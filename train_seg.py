import torch
from torch.optim import SGD, Adam, lr_scheduler
from tqdm import tqdm
import math
from torch.cuda import amp
import torch
from utils.loss import BinaryDiceLoss
import torch.nn as nn
import yaml
from basemodel import TextDetector
import numpy as np
from datetime import datetime
from torchsummary import summary
import numexpr
import os
import shutil
os.environ['NUMEXPR_MAX_THREADS'] = str(numexpr.detect_number_of_cores())

from seg_dataset import create_dataloader
from utils.general import LOGGER, Loggers, CUDA, DEVICE
import random

torch.random.manual_seed(0)
random.seed(0)
np.random.seed(0)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

def eval_model(model: nn.Module, val_loader):
    global DEVICE
    loss_func = BinaryDiceLoss()
    pbar = enumerate(val_loader)
    nb = len(val_loader)
    model.eval()
    pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    pr = tp = gt = m_loss = 0
    with torch.no_grad():
        for i, (imgs, masks) in pbar:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            pred = model(imgs)
            imgs.detach_()
            del imgs
            tp += torch.mul(pred, masks).sum().detach_()
            gt += masks.sum().detach_()
            pr += pred.sum().detach_()
            loss = loss_func(pred, masks)
            m_loss = (m_loss * i + loss.detach()) / (i + 1)
            masks.detach_()
            del masks
    recall = tp / gt
    precision = tp / pr
    return recall, precision, m_loss

def train(hyp):
    with open(r'data/training_hyp.yaml', 'w', encoding='utf8') as f:
        yaml.safe_dump(hyp, f)
    start_epoch = 0
    hyp_train, hyp_data, hyp_model, hyp_logger, hyp_resume = hyp['train'], hyp['data'], hyp['model'], hyp['logger'], hyp['resume']
    epochs = hyp_train['epochs']
    batch_size = hyp_train['batch_size']
    model = TextDetector(**hyp_model)
    if CUDA:
        model.cuda()
    params = model.seg_net.parameters()
    
    if hyp_train['optimizer'] == 'adam':
        optimizer = Adam(params, lr=hyp_train['lr0'], betas=(hyp_train['momentum'], 0.999), weight_decay=hyp_train['weight_decay'])  # adjust beta1 to momentum
    else:
        optimizer = SGD(params, lr=hyp_train['lr0'], momentum=hyp_train['momentum'], nesterov=True, weight_decay=hyp_train['weight_decay'])
    
    if hyp_train['linear_lr']:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp_train['lrf']) + hyp_train['lrf']  # linear
    else:
        lf = one_cycle(1, hyp_train['lrf'], epochs)  # cosine 1->hyp['lrf']

    scaler = amp.GradScaler(enabled=CUDA)
    loss_func = BinaryDiceLoss()

    # Scheduler
    if hyp_train['linear_lr']:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp_train['lrf']) + hyp_train['lrf']  # linear
    else:
        lf = one_cycle(1, hyp_train['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    
    logger = None
    if hyp_resume['resume_training']:
        LOGGER.info(f'resume traning ... ')
        ckpt = torch.load(hyp_resume['ckpt'], map_location=DEVICE)
        model.seg_net.load_state_dict(ckpt['weights'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scheduler.step()
        start_epoch = ckpt['epoch'] + 1
        hyp_logger['run_id'] = ckpt['run_id']
        logger = Loggers(hyp)

    else:
        if hyp_logger['type'] == 'wandb':
            logger = Loggers(hyp)

    num_workers = 8
    train_img_dir, train_mask_dir, imgsz, augment, aug_param = hyp_data['train_img_dir'], hyp_data['train_mask_dir'], hyp_data['imgsz'], hyp_data['augment'], hyp_data['aug_param']
    val_img_dir, val_mask_dir = hyp_data['val_img_dir'], hyp_data['val_mask_dir']
    train_dataset, train_loader = create_dataloader(train_img_dir, train_mask_dir, imgsz, batch_size, augment, aug_param, shuffle=True, workers=num_workers, cache=hyp_data['cache'])
    val_dataset, val_loader = create_dataloader(val_img_dir, val_mask_dir, imgsz, 4, augment=False, shuffle=False, workers=num_workers, cache=hyp_data['cache'])
    nb = len(train_loader)
    nw = max(round(3 * nb), 700)

    LOGGER.info(f'num training imgs: {len(train_dataset)}, num val imgs: {len(val_dataset)}')

    eval_interval = hyp_train['eval_interval']
    best_f1 = -1
    best_val_loss = np.inf
    accumulation_steps = hyp_train['accumulation_steps']
    summary(model, (3, 640, 640), device=DEVICE)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        
        model.train_mask()
        train_dataset.initialize() 
        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        
        m_loss = 0
        for i, (imgs, masks) in pbar:
            
            pbar.set_description(f' training size: {train_dataset.img_size}')
            # warm up
            ni = i + nb * epoch
            if ni <= nw:
                xi = [0, nw]  # x interp
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp_train['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp_train['warmup_momentum'], hyp_train['momentum']])

            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            with amp.autocast():
                preds = model(imgs)
                imgs.detach_()
                del imgs
                loss = loss_func(preds, masks)
                masks.detach_()
                del masks
            scaler.scale(loss).backward()
            if i % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            m_loss = (m_loss * i + loss.detach()) / (i + 1)
        
        if (epoch + 1) % eval_interval == 0:
            recall, precision, eval_m_loss = eval_model(model, val_loader)
            f1 = 2 * recall * precision / (recall + precision)
            last_ckpt = {'epoch': epoch,
                        'best_f1': best_f1,
                        'weights': model.seg_net.state_dict(),
                        'best_val_loss': best_val_loss,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'run_id': logger.wandb.id if logger is not None else None,
                        'date': datetime.now().isoformat(),
                        'hyp': hyp}
            torch.save(last_ckpt, 'data/unet_last.ckpt')
            if best_f1 < f1:
                best_f1 = f1
                LOGGER.info(f'saving model at epoch {epoch}, best val f1: {best_f1}')
                shutil.copy2('data/unet_last.ckpt', 'data/unet_best.ckpt')
            LOGGER.info(f'epoch {epoch}/{epochs-1} loss: {m_loss} precision: {precision} recall: {recall}')
            if logger is not None:
                log_dict = {}
                log_dict['train/lr'] = optimizer.param_groups[0]['lr']
                log_dict['train/loss'] = m_loss
                log_dict['eval/recall'] = recall
                log_dict['eval/precision'] = precision
                log_dict['eval/f1'] = f1
                log_dict['eval/eval_m_loss'] = eval_m_loss
                logger.on_train_epoch_end(epoch, log_dict)
        scheduler.step()
        pbar.close()

if __name__ == '__main__':
    hyp_p = r'data/train_hyp.yaml'
    with open(hyp_p, 'r', encoding='utf8') as f:
        hyp = yaml.safe_load(f.read())

    hyp['data']['train_img_dir'] = [r'../datasets/codat_manga_v3/images/train', r'../datasets/ComicErased/processed']
    # hyp['data']['train_img_dir'] = [r'../datasets/codat_manga_v3/images/val']
    hyp['data']['val_img_dir'] = [r'../datasets/codat_manga_v3/images/val']
    hyp['data']['train_mask_dir'] = r'../datasets/ComicSegV2'
    hyp['data']['val_mask_dir'] = r'../datasets/ComicSegV2'
    hyp['data']['imgsz'] = 1024
    hyp['data']['cache'] = False
    hyp['data']['aug_param']['neg'] = 0.3
    hyp['data']['aug_param']['size_range'] = [0.85, 1.1]

    hyp['train']['lr0'] = 0.004
    hyp['train']['lrf'] = 0.005
    hyp['train']['weight_decay'] = 0.00002
    hyp['train']['epochs'] = 120
    hyp['train']['accumulation_steps'] = 4
    hyp['train']['batch_size'] = 4
    hyp['logger']['type'] = 'wandb'

    # hyp['resume']['resume_training'] = True
    # hyp['resume']['ckpt'] = 'data/unet_last.ckpt'
    train(hyp)