import os
import os.path as osp
import glob
from pathlib import Path
import cv2
import numpy as np

IMG_EXT = ['.bmp', '.jpg', '.png', '.jpeg']

def find_all_imgs(img_dir, abs_path=False):
    imglist = list()
    for filep in glob.glob(osp.join(img_dir, "*")):
        filename = osp.basename(filep)
        file_suffix = Path(filename).suffix
        if file_suffix.lower() not in IMG_EXT:
            continue
        if abs_path:
            imglist.append(filep)
        else:
            imglist.append(filename)
    return imglist

def imread(imgpath, read_type=cv2.IMREAD_COLOR):
    # img = cv2.imread(imgpath, read_type)
    # if img is None:
    img = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), read_type)
    return img

def imwrite(img_path, img, ext='.png'):
    suffix = Path(img_path).suffix
    if suffix != '':
        img_path = img_path.replace(suffix, ext)
    else:
        img_path += ext
    cv2.imencode(ext, img)[1].tofile(img_path)