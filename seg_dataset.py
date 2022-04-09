import glob
import os
import os.path as osp
import random
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from zipfile import ZipFile

import cv2
import numpy as np
from numpy.lib.npyio import load
from numpy.random import rand
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision.transforms.transforms import Compose

from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path

from tqdm import tqdm

from torchvision import transforms
import random
from torch.utils.data import DataLoader, Dataset
from utils.general import LOGGER, Loggers, CUDA, DEVICE
from utils.imgproc_utils import resize_keepasp, letterbox
from utils.io_utils import imread, imwrite

WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # DPP
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of multiprocessing threads
IMG_EXT = ['.bmp', '.jpg', '.png', '.jpeg']

def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed

def load_image_mask(self, i, max_size=None):
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    img, mask = self.imgs[i], self.masks[i]
    imp, maskp = self.img_mask_list[i]
    if img is None:
        img = cv2.imread(imp)
    if mask is None:
        mask = cv2.imread(maskp, cv2.IMREAD_GRAYSCALE)
    if max_size is not None:
        if isinstance(max_size, tuple):
            max_size = max_size[0]
        try:
            img = resize_keepasp(img, max_size)
            mask = resize_keepasp(mask, max_size, interpolation=cv2.INTER_AREA)
        except:
            pass
    return img, mask

def mini_mosaic(self, img, mask):
    im_h, im_w = img.shape[0], img.shape[1]
    idx = random.randint(0, len(self)-1)
    img2, mask2 = load_image_mask(self, idx, self.img_size)
    img2_h, img2_w = img2.shape[0], img2.shape[1]
    ratio = img2_h / im_h
    if img2_h > img2_w and ratio > 0.4 and ratio < 1.6:
        im_h = max(im_h, img2_h)
        im_w = im_w + img2_w
        im_tmp = np.zeros((im_h, im_w, 3), np.uint8)
        im_tmp[:img.shape[0], :img.shape[1]] = img
        im_tmp[:img2_h, img.shape[1]:] = img2
        mask_tmp = np.zeros((im_h, im_w), np.uint8)
        mask_tmp[:img.shape[0], :img.shape[1]] = mask
        mask_tmp[:img2_h, img.shape[1]:] = mask2

        img = np.ascontiguousarray(im_tmp)
        mask = np.ascontiguousarray(mask_tmp)
    return img, mask

class LoadImageAndMask(Dataset):
    def __init__(self, img_dir, mask_dir=None, img_size=640, augment=False, aug_param=None, cache=False, stride=128, cache_mask_only=True):
        if isinstance(img_dir, str):
            self.img_dir = [img_dir]
        elif isinstance(img_dir, list):
            self.img_dir = img_dir
        else:
            raise Exception('unknown img_dir format')
        
        if mask_dir is None or mask_dir == '':
            self.mask_dir = self.img_dir
        else:
            if isinstance(mask_dir, str):
                self.mask_dir = [mask_dir]
            elif isinstance(mask_dir, list):
                self.mask_dir = mask_dir
        
        self.img_mask_list = []
        self.img_size = (img_size, img_size)
        self.stride = stride
        self._augment = augment
        if self._augment:
            self._mini_mosaic = aug_param['mini_mosaic']
            self._augment_hsv = aug_param['hsv']
            self._flip_lr = aug_param['flip_lr']
            self._neg = aug_param['neg']
            size_range = aug_param['size_range'] 
            if size_range[0] != -1:
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
                if file_suffix.lower() not in IMG_EXT:
                    continue
                maskname = 'mask-' + filename.replace(file_suffix, '.png')
                for mask_dir in self.mask_dir:
                    maskp = osp.join(mask_dir, maskname)
                    if osp.exists(maskp):
                        self.img_mask_list.append((filep, maskp))
        self._img_transform = transforms.Compose([transforms.ToTensor()])
        self._mask_transform = transforms.Compose([transforms.ToTensor()])

        n = len(self.img_mask_list)
        self.imgs, self.masks = [None] * n, [None] * n
        gb = 0
        if cache:
            results = ThreadPool(NUM_THREADS).imap(lambda x: load_image_mask(*x, max_size=img_size), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                im, self.masks[i] = x  # im, hw_orig, hw_resized = load_image_mask(self, i)
                if not cache_mask_only:
                    self.imgs[i] = im
                    gb += self.imgs[i].nbytes
                gb += self.masks[i].nbytes
                if gb / 1E9 > 7:
                    break
                pbar.desc = f'Caching images ({gb / 1E9:.1f}GB )'
            pbar.close()
        
    def initialize(self):
        if self.augment:
            if self.multi_size:
                self.img_size = random.choice(self.valid_size)
    
    def transform(self, img, mask):
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        img = img.astype(np.float32) / 255
        mask = (mask > 30).astype(np.float32)
        # mask = mask / 255
        img = self._img_transform(img)
        mask = self._mask_transform(mask)
        return img, mask

    def augment(self, img, mask):
        im_h, im_w = img.shape[0], img.shape[1]
        if im_h > im_w and random.random() < self._mini_mosaic:
            # imp2, maskp2 = random.choice(self.img_mask_list)
            img, mask = mini_mosaic(self, img, mask)

        img, ratio, (dw, dh) = letterbox(img, new_shape=self.img_size, auto=False)
        mask, ratio, (dw, dh) = letterbox(mask, new_shape=self.img_size, auto=False)

        if random.random() < self._augment_hsv:
            augment_hsv(img)
        if random.random() < self._flip_lr:
            cv2.flip(img, 1, img)
            cv2.flip(mask, 1, mask)
        if random.random() < self._neg:
            img = 255 - img
        return img, mask

    def inverse_transform(self, img: torch.Tensor):
        img = img.permute(1, 2, 0)
        img = img * 255
        img = img.cpu().numpy().astype(np.uint8)
        return img

    def __len__(self):
        return len(self.img_mask_list)

    def __getitem__(self, idx):
        img, mask = load_image_mask(self, idx, self.img_size)
        if self._augment:
            img, mask = self.augment(img, mask)
        else:
            img, ratio, (dw, dh) = letterbox(img, new_shape=self.img_size, auto=False)
            mask, ratio, (dw, dh) = letterbox(mask, new_shape=self.img_size, auto=False)
        return self.transform(img, mask)

def create_dataloader(img_dir, mask_dir, imgsz, batch_size, augment=False, aug_param=None, cache=False, workers=8, shuffle=False):
    dataset = LoadImageAndMask(img_dir, mask_dir, imgsz, augment, aug_param, cache)
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // WORLD_SIZE, batch_size if batch_size > 1 else 0, workers])  # number of workers
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=nw)
    return dataset, loader

if __name__ == '__main__':
    random.seed(42)
    torch.random.manual_seed(42)
    np.random.seed(42)
    import yaml
    hyp_p = r'data/train_hyp.yaml'
    with open(hyp_p, 'r', encoding='utf8') as f:
        hyp = yaml.safe_load(f.read())
    hyp['data']['train_img_dir'] = [r'D:/neonbub/datasets/codat_manga_v3/images/train', r'D:/neonbub/datasets/ComicErased/processed']
    hyp['data']['val_img_dir'] = [r'D:/neonbub/datasets/codat_manga_v3/images/val']
    hyp['data']['train_mask_dir'] = r'D:/neonbub/datasets/ComicSegV2'
    hyp['data']['val_mask_dir'] = r'D:/neonbub/datasets/ComicSegV2'
    hyp['data']['cache'] = False

    hyp_train, hyp_data, hyp_model, hyp_logger, hyp_resume = hyp['train'], hyp['data'], hyp['model'], hyp['logger'], hyp['resume']
    
    batch_size = hyp_train['batch_size']
    batch_size = 4
    num_workers = 2

    train_img_dir, train_mask_dir, imgsz, augment, aug_param = hyp_data['train_img_dir'], hyp_data['train_mask_dir'], hyp_data['imgsz'], hyp_data['augment'], hyp_data['aug_param']
    val_img_dir, val_mask_dir = hyp_data['val_img_dir'], hyp_data['val_mask_dir']
    train_dataset, train_loader = create_dataloader(train_img_dir, train_mask_dir, imgsz, batch_size, augment, aug_param, shuffle=True, workers=num_workers, cache=hyp_data['cache'])
    val_dataset, val_loader = create_dataloader(val_img_dir, val_mask_dir, imgsz, batch_size, augment=False, shuffle=False, workers=num_workers, cache=hyp_data['cache'])
    LOGGER.info(f'num training imgs: {len(train_dataset)}, num val imgs: {len(val_dataset)}')
    
    for epoch in range(0, 4):  # epoch ------------------------------------------------------------------
        train_dataset.initialize()
        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=len(train_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        pbar.set_description(f' training size: {train_dataset.img_size}')
        for i, (imgs, masks) in pbar:
            img, mask = imgs[0], masks[0]
            imgs = imgs
            masks = masks
            img = train_dataset.inverse_transform(img)
            mask = train_dataset.inverse_transform(mask)
            cv2.imshow('img', img)
            cv2.imshow('mask', mask)
            cv2.waitKey(0)
        pbar.close()