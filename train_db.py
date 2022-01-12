from torch.autograd.grad_mode import F
from torch.nn.functional import sigmoid
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import SGD, Adam, lr_scheduler
from tqdm import tqdm
import math
from torch.cuda import amp
import torch
from loss import DBLoss
import torch.nn as nn
import yaml
from basemodel import TextDetector
from utils.db_utils import SegDetectorRepresenter, QuadMetric
import numpy as np
from datetime import datetime
from torchsummary import summary
import numexpr
import os
import shutil
os.environ['NUMEXPR_MAX_THREADS'] = str(numexpr.detect_number_of_cores())

from db_dataset import create_dataloader
from utils.general import LOGGER, Loggers, CUDA, DEVICE
import time
import random

torch.random.manual_seed(0)
random.seed(0)
np.random.seed(0)

def one_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

def eval_model(model: nn.Module, val_loader, post_process, metric_cls):
    # global DEVICE
    raw_metrics = []
    total_frame = 0.0
    total_time = 0.0
    model.eval()
    for i, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc='test model'):
        with torch.no_grad():
            # 数据进行转换和丢到gpu
            for key, value in batch.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(DEVICE)
            start = time.time()
            with amp.autocast():
                preds = model(batch['imgs'])
            boxes, scores = post_process(batch, preds,is_output_polygon=False)
            total_frame += batch['imgs'].size()[0]
            total_time += time.time() - start
            raw_metric = metric_cls.validate_measure(batch, (boxes, scores))
            raw_metrics.append(raw_metric)
    metrics = metric_cls.gather_measure(raw_metrics)
    LOGGER.info('FPS:{}'.format(total_frame / total_time))
    return metrics['recall'].avg, metrics['precision'].avg, metrics['fmeasure'].avg

def train(hyp):
    start_epoch = 0
    hyp_train, hyp_data, hyp_model, hyp_logger, hyp_resume = hyp['train'], hyp['data'], hyp['model'], hyp['logger'], hyp['resume']
    epochs = hyp_train['epochs']
    batch_size = hyp_train['batch_size']

    scaler = amp.GradScaler(enabled=CUDA)
    criterion = DBLoss()
    use_bce = False
    if hyp_train['loss'] == 'bce':
        use_bce = True
    shrink_with_sigmoid = not use_bce

    model = TextDetector(hyp_model['weights'], map_location='cpu')
    model.initialize_db(hyp_model['unet_weights'])
    model.dbnet.shrink_with_sigmoid = shrink_with_sigmoid
    model.train_db()
    model.to(DEVICE)

    if hyp_model['db_weights'] != '':
        model.dbnet.load_state_dict(torch.load(hyp_model['db_weights'])['weights'])
    if hyp_train['optimizer'] == 'adam': 
        optimizer = Adam(model.dbnet.parameters(), lr=hyp_train['lr0'], betas=(0.937, 0.999), weight_decay=0.00002)  # adjust beta1 to momentum
    else:
        optimizer = SGD(model.dbnet.parameters(), lr=hyp_train['lr0'], momentum=hyp_train['momentum'], nesterov=True, weight_decay=hyp_train['weight_decay'])
    
    if hyp_train['linear_lr']:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp_train['lrf']) + hyp_train['lrf']  # linear
    else:
        lf = one_cycle(1, hyp_train['lrf'], epochs)  # cosine 1->hyp['lrf']

    if hyp_train['linear_lr']:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp_train['lrf']) + hyp_train['lrf']  # linear
    else:
        lf = one_cycle(1, hyp_train['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    
    logger = None
    if hyp_resume['resume_training']:
        LOGGER.info(f'resume traning ... ')
        ckpt = torch.load(hyp_resume['ckpt'], map_location=DEVICE)
        model.dbnet.load_state_dict(ckpt['weights'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scheduler.step()
        start_epoch = ckpt['epoch'] + 1
        hyp_logger['run_id'] = ckpt['run_id']
        logger = Loggers(hyp)

    else:
        # if hyp_logger['type'] == 'wandb':
        logger = Loggers(hyp)

    num_workers = 8
    train_img_dir, train_mask_dir, imgsz, augment, aug_param = hyp_data['train_img_dir'], hyp_data['train_mask_dir'], hyp_data['imgsz'], hyp_data['augment'], hyp_data['aug_param']
    val_img_dir, val_mask_dir = hyp_data['val_img_dir'], hyp_data['val_mask_dir']
    train_dataset, train_loader = create_dataloader(train_img_dir, train_mask_dir, imgsz, batch_size, augment, aug_param, shuffle=True, workers=num_workers, cache=hyp_data['cache'])
    val_dataset, val_loader = create_dataloader(val_img_dir, val_mask_dir, imgsz, batch_size, augment=False, shuffle=False, workers=num_workers, cache=hyp_data['cache'], with_ann=True)
    nb = len(train_loader)
    nw = max(round(3 * nb), 700)

    LOGGER.info(f'num training imgs: {len(train_dataset)}, num val imgs: {len(val_dataset)}')

    eval_interval = hyp_train['eval_interval']
    best_f1 = best_epoch = -1
    best_val_loss = np.inf

    accumulation_steps = hyp_train['accumulation_steps']
    # summary(model, (3, 640, 640), device=DEVICE)
    metric_cls = QuadMetric()
    post_process = SegDetectorRepresenter(thresh=0.5)
    best_f1 = -1
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        train_dataset.initialize()
        model.train_db()
        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        pbar.set_description(f' training size: {train_dataset.img_size}')
        m_loss = 0
        m_loss_s = 0
        m_loss_t = 0
        m_loss_b = 0
        for i, batchs in pbar:
            # warm up
            if hyp_train['warm_up']:
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    for j, x in enumerate(optimizer.param_groups):
                        x['lr'] = np.interp(ni, xi, [hyp_train['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [hyp_train['warmup_momentum'], hyp_train['momentum']])

            with amp.autocast():
                for key in batchs.keys():
                    batchs[key] = batchs[key].cuda()
                preds = model(batchs['imgs'])
                metric = criterion(preds, batchs, use_bce)
            loss = metric['loss'] / accumulation_steps
            scaler.scale(loss).backward()
            if i % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            m_loss = (m_loss * i + metric['loss'].detach()) / (i + 1)
            m_loss_s = (m_loss_s * i + metric['loss_shrink_maps'].detach()) / (i + 1)
            m_loss_t = (m_loss_t * i + metric['loss_threshold_maps'].detach()) / (i + 1)
            m_loss_b = (m_loss_b * i + metric['loss_binary_maps'].detach()) / (i + 1)

        if i % eval_interval == 0:
            recall, precision, fmeasure = eval_model(model,  val_loader, post_process, metric_cls)
            log_dict = {}
            log_dict['train/lr'] = optimizer.param_groups[0]['lr']
            log_dict['train/loss'] = m_loss
            log_dict['train/loss_shrink'] = m_loss_s
            log_dict['train/loss_threshold'] = m_loss_t
            log_dict['train/loss_binary_maps'] = m_loss_b
            log_dict['eval/recall'] = recall
            log_dict['eval/precision'] = precision
            log_dict['eval/f1'] = fmeasure
            
            save_best = best_f1 < fmeasure
            if save_best:
                best_f1 = fmeasure
            last_ckpt = {'epoch': epoch,
                        'best_f1': best_f1,
                        'weights': model.dbnet.state_dict(),
                        'best_val_loss': best_val_loss,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'run_id': logger.wandb.id if logger.wandb is not None else None,
                        'date': datetime.now().isoformat(),
                        'hyp': hyp}
            torch.save(last_ckpt, 'data/db_last.pt')
            if save_best:
                shutil.copy('data/db_last.pt', 'data/db_best.pt')
            if logger is not None:
                logger.on_train_epoch_end(epoch, log_dict)
        scheduler.step()
        pbar.close()

if __name__ == '__main__':
    hyp_p = r'data/train_db_hyp.yaml'
    with open(hyp_p, 'r', encoding='utf8') as f:
        hyp = yaml.safe_load(f.read())

    # hyp['data']['train_img_dir'] = r'../datasets/pixanimegirls/processed'
    hyp['data']['train_img_dir'] = [r'../datasets/codat_manga_v3/images/train', r'../datasets/codat_manga_v3/images/val', r'../datasets/pixanimegirls/processed']
    hyp['data']['train_mask_dir'] = r'../datasets/TextLines'
    # hyp['data']['train_img_dir'] = r'data/dataset/db_sub'
    hyp['data']['val_img_dir'] = r'data/dataset/db_sub'
    hyp['data']['cache'] = False
    # hyp['data']['aug_param']['size_range'] = [-1]

    hyp['train']['lr0'] = 0.01
    hyp['train']['lrf'] = 0.005
    hyp['train']['weight_decay'] = 0.00002
    hyp['train']['batch_size'] = 4
    hyp['train']['epochs'] = 100
    # hyp['train']['optimizer'] = 'sgd'

    hyp['train']['loss'] = 'bce'
    hyp['logger']['type'] =  None

    hyp['resume']['resume_training'] = False
    hyp['resume']['ckpt'] = 'last.pt'
    train(hyp)