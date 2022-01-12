import numpy as np
import yaml
import torch
import glob
import os
import os.path as osp
import random
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
import cv2
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, dataloader
from utils.general import LOGGER, Loggers, CUDA, DEVICE
from utils.db_utils import MakeBorderMap, MakeShrinkMap
from dataset import letterbox, augment_hsv, resize_keepasp

WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # DPP
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of multiprocessing threads
IMG_EXT = ['.bmp', '.jpg', '.png', '.jpeg']

def db_val_collate_fn(batchs):
    cat_list = ['text_polys', 'ignore_tags']
    ret_batchs = {}
    for key in batchs[0].keys():
        ret_batchs[key] = []
        for batch in batchs:
            if isinstance(batch[key], np.ndarray):
                batch[key] = torch.from_numpy(batch[key])
            ret_batchs[key].append(batch[key])
        if key in cat_list:
            pass
        else:
            ret_batchs[key] = torch.stack(ret_batchs[key], 0)
    return ret_batchs

class LoadImageAndAnnotations(Dataset):
    def __init__(self, img_dir, ann_dir=None, img_size=640, augment=False, aug_param=None, cache=False, stride=128, cache_ann_only=True, with_ann=False):
        if isinstance(img_dir, str):
            self.img_dir = [img_dir]
        elif isinstance(img_dir, list):
            self.img_dir = img_dir
        else:
            raise Exception('unknown img_dir format')
        
        if ann_dir is None or ann_dir == '':
            self.ann_dir = self.img_dir
        else:
            if isinstance(ann_dir, str):
                self.ann_dir = [ann_dir]
            elif isinstance(ann_dir, list):
                self.ann_dir = ann_dir
        self.with_ann = with_ann
        self.make_border_map = MakeBorderMap(shrink_ratio=0.4)
        self.make_shrink_map = MakeShrinkMap(shrink_ratio=0.4)
        self.img_ann_list = []
        self.img_size = (img_size, img_size)
        self.stride = stride
        self._augment = augment
        if self._augment:
            self._mini_mosaic = aug_param['mini_mosaic']
            self._augment_hsv = aug_param['hsv']
            self._flip_lr = aug_param['flip_lr']
            self._neg = aug_param['neg']
            size_range = aug_param['size_range'] 
            if isinstance(size_range, list) and size_range[0] > 0:
                min_size = round(img_size * size_range[0] / stride ) * stride
                max_size = round(img_size * size_range[1] / stride ) * stride
                self.valid_size = np.arange(min_size, max_size+1, stride)
                self.multi_size = True
            else:
                self.valid_size = None
                self.multi_size = False
        for img_dir in self.img_dir:
            for filep in glob.glob(osp.join(img_dir, "*")):
                filename = osp.basename(filep)
                file_suffix = Path(filename).suffix
                if file_suffix not in IMG_EXT:
                    continue
                annname = 'line-' + filename.replace(file_suffix, '.txt')
                for ann_dir in self.ann_dir:
                    annp = osp.join(ann_dir, annname)
                    if osp.exists(annp):
                        self.img_ann_list.append((filep, annp))
        self._img_transform = transforms.Compose([transforms.ToTensor()])

        n = len(self.img_ann_list)
        self.imgs, self.anns = [None] * n, [None] * n
        gb = 0
        if cache:
            results = ThreadPool(NUM_THREADS).imap(lambda x: load_image_annotations(*x, max_size=img_size), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                im, self.anns[i] = x  # im, hw_orig, hw_resized = load_image_ann(self, i)
                if not cache_ann_only:
                    self.imgs[i] = im
                    gb += self.imgs[i].nbytes
                gb += self.anns[i].nbytes
                if gb / 1E9 > 7:
                    break
                pbar.desc = f'Caching images ({gb / 1E9:.1f}GB )'
            pbar.close()
        
    def initialize(self):
        if self.augment:
            if self.multi_size:
                self.img_size = random.choice(self.valid_size)
    
    def transform(self, img):
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        img = img.astype(np.float32) / 255
        img = self._img_transform(img)
        return img

    def mini_mosaic(self, img, ann):
        im_h, im_w = img.shape[:2]
        idx = random.randint(0, len(self)-1)
        img2, ann2 = load_image_annotations(self, idx, self.img_size)
        img2_h, img2_w = img2.shape[:2]
        
        if img2_h > img2_w:
            imm_h = max(im_h, img2_h)
            imm_w = im_w + img2_w
            im_tmp = np.zeros((imm_h, imm_w, 3), np.uint8)
            im_tmp[:im_h, :im_w] = img
            im_tmp[:img2_h, im_w:] = img2
            ann[:, :, 0] = ann[:, :, 0] * im_w / imm_w
            ann[:, :, 1] = ann[:, :, 1] * im_h / imm_h
            if ann2.shape[1] > 0:
                ann2[:, :, 0] = ann2[:, :, 0] * img2_w / imm_w + im_w / imm_w
                ann2[:, :, 1] = ann2[:, :, 1] * img2_h / imm_h
                ann = np.concatenate((ann, ann2))
            img = im_tmp
            return img, ann
                
        else:
            return img, ann

    def augment(self, img, ann):
        im_h, im_w = img.shape[0], img.shape[1]
        if im_h > im_w and random.random() < self._mini_mosaic:
            # imp2, annp2 = random.choice(self.img_ann_list)
            img, ann = self.mini_mosaic(img, ann)

        if random.random() < self._augment_hsv:
            augment_hsv(img)
        if random.random() < self._flip_lr:
            cv2.flip(img, 1, img)
            ann[:, :, 0] = 1 - ann[:, :, 0]
        if random.random() < self._neg:
            img = 255 - img
        return img, ann

    def inverse_transform(self, img: torch.Tensor, scale=255, to_uint8=True):
        img = img.permute(1, 2, 0)
        img = img * scale
        img = img.cpu().numpy()
        if to_uint8:
            img = np.ascontiguousarray(img, np.uint8)
        return img

    def __len__(self):
        return len(self.img_ann_list)

    def __getitem__(self, idx):
        img, ann = load_image_annotations(self, idx, self.img_size)
        in_h, in_w = img.shape[:2]

        if self._augment:
            img, ann = self.augment(img, ann)
        ignore_tags = [False] * ann.shape[0]

        img, ratio, (dw, dh) = letterbox(img, new_shape=self.img_size, auto=False)
        im_h, im_w = img.shape[:2]
        if ann is not None:
            ann[:, :, 0] *= (im_w - dw)
            ann[:, :, 1] *= (im_h - dh)
            ann = ann.astype(np.int64)
        data_dict = {'imgs': img, 'text_polys': ann, 'ignore_tags': ignore_tags}

        shrink_map = self.make_shrink_map(data_dict)
        thresh_map = self.make_border_map(data_dict)
        tp = thresh_map.pop('text_polys')
        it = thresh_map.pop('ignore_tags')
        if self.with_ann:
            thresh_map['text_polys'] = torch.from_numpy(np.array(tp))
            thresh_map['ignore_tags'] = torch.from_numpy(np.array(it))

        thresh_map['imgs'] = self.transform(thresh_map['imgs'])
        return thresh_map


def load_image_annotations(self, i, max_size=None, ann_abs2rel=True):
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    img, ann = self.imgs[i], self.anns[i]
    imp, ann_path = self.img_ann_list[i]
    if img is None:
        img = cv2.imread(imp)
    im_h, im_w = img.shape[:2]
    if ann is None:
        ann = np.loadtxt(ann_path)
        if len(ann.shape) == 1:
            ann = np.array([ann])
        if ann_abs2rel:
            ann[:, ::2] /= im_w
            ann[:, 1::2] /= im_h
        ann = ann.reshape(len(ann), -1, 2)
    else:
        ann = np.copy(ann)
    if max_size is not None:
        if isinstance(max_size, tuple):
            max_size = max_size[0]
        img = resize_keepasp(img, max_size)
    return img, ann

def create_dataloader(img_dir, ann_dir, imgsz, batch_size, augment=False, aug_param=None, cache=False, workers=8, shuffle=False, with_ann=False):
    dataset = LoadImageAndAnnotations(img_dir, ann_dir, imgsz, augment, aug_param, cache, with_ann=with_ann)
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // WORLD_SIZE, batch_size if batch_size > 1 else 0, workers])  # number of workers
    if with_ann:
        collate_fn = db_val_collate_fn
    else:
        collate_fn = None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=nw, collate_fn=collate_fn)
    return dataset, loader

if __name__ == '__main__':
    img_dir = r'dataset/db_sub'
    hyp_p = r'data/train_db_hyp.yaml'
    with open(hyp_p, 'r', encoding='utf8') as f:
        hyp = yaml.safe_load(f.read())
    hyp['data']['train_img_dir'] = img_dir
    hyp['data']['cache'] = False
    hyp_train, hyp_data, hyp_model, hyp_logger, hyp_resume = hyp['train'], hyp['data'], hyp['model'], hyp['logger'], hyp['resume']
    batch_size = hyp_train['batch_size']
    batch_size = 4
    num_workers = 0
    train_img_dir, train_mask_dir, imgsz, augment, aug_param = hyp_data['train_img_dir'], hyp_data['train_mask_dir'], hyp_data['imgsz'], hyp_data['augment'], hyp_data['aug_param']

    train_dataset, train_loader = create_dataloader(train_img_dir, train_mask_dir, imgsz, batch_size, augment, aug_param, shuffle=True, workers=num_workers, cache=hyp_data['cache'], with_ann=True)

    for ii in range(10):
        train_dataset.initialize()
        for batchs in train_loader:
            img = batchs['imgs'][0]
            
            img = train_dataset.inverse_transform(img)
            threshold_map = batchs['threshold_map'][0]
            threshold_mask = batchs['threshold_mask'][0]
            shrink_map = batchs['shrink_map'][0]
            shrink_mask = batchs['shrink_mask'][0]
            polys = batchs['text_polys'][0].numpy().astype(np.int32)
            for p in polys:
                cv2.polylines(img,[p],True,(255, 0, 0), thickness=2)
            cv2.imshow('imgs', img)
            cv2.imshow('threshold_map', threshold_map.numpy())
            cv2.imshow('threshold_mask', threshold_mask.numpy())
            cv2.imshow('shrink_map', shrink_map.numpy())
            cv2.imshow('shrink_mask', shrink_mask.numpy())
            cv2.waitKey(0) 