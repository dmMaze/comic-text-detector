
from importlib.resources import path
from basemodel import UnetHead, DBHead, TextDetBase
from utils.yolov5_utils import non_max_suppression
from models.yolo import load_yolov5
from collections import OrderedDict
import os.path as osp
import os
from tqdm import tqdm
import numpy as np
import cv2
import torch
from utils.db_utils import SegDetectorRepresenter
from dataset import letterbox
from pathlib import Path
import glob
import torch

IMG_EXT = ['.bmp', '.jpg', '.png', '.jpeg']

def xywh2xyxypoly(xywh):
    xyxypoly = np.tile(xywh[:, [0, 1]], 4)
    xyxypoly[:, [2, 4]] += xywh[:, [2]]
    xyxypoly[:, [5, 7]] += xywh[:, [3]]
    return xyxypoly.astype(np.int64)

def union_area(bboxa, bboxb):
    x1 = max(bboxa[0], bboxb[0])
    y1 = max(bboxa[1], bboxb[1])
    x2 = min(bboxa[2], bboxb[2])
    y2 = min(bboxa[3], bboxb[3])
    if y2 < y1 or x2 < x1:
        return -1
    return (y2 - y1) * (x2 - x1)

def yolo_xywh2xyxy(xywh: np.array, w: int, h:  int):
    if xywh is None:
        return None
    if len(xywh) == 0:
        return None
    if len(xywh.shape) == 1:
        xywh = np.array([xywh])
    xywh[:, [0, 2]] *= w
    xywh[:, [1, 3]] *= h
    xywh[:, [0, 1]] -= xywh[:, [2, 3]] / 2
    xywh[:, [2, 3]] += xywh[:, [0, 1]]
    return xywh.astype(np.int64)

def xyxy2yolo(xyxy, w: int, h: int, ordered=False):
    if xyxy == [] or xyxy == np.array([]) or len(xyxy) == 0:
        return None
    if isinstance(xyxy, list):
        xyxy = np.array(xyxy)
    if len(xyxy.shape) == 1:
        xyxy = np.array([xyxy])
    yolo = np.copy(xyxy).astype(np.float64)
    yolo[:, [0, 2]] =  yolo[:, [0, 2]] / w
    yolo[:, [1, 3]] = yolo[:, [1, 3]] / h
    yolo[:, [2, 3]] -= yolo[:, [0, 1]]
    yolo[:, [0, 1]] += yolo[:, [2, 3]] / 2
    return yolo

def xywh2yolo(xywh: np.array, w: int, h: int):
    if len(xywh.shape) == 1:
        xywh = np.array([xywh])
    xywh = np.copy(xywh)
    xywh[:, [0, 2]] /= w
    xywh[:, [1, 3]] /= h
    xywh[:, [0, 1]] += xywh[:, [2, 3]] / 2
    return xywh



def grid_sort(pred, im_w, im_h):
    # sort textblock by 3*3 or 3*4 "grid"
    if im_h > im_w:
        ghNum, gwNum = 4, 3
    else: ghNum, gwNum = 3, 3
    border_tol, center_tol = 10, (min(im_h, im_w) / 10)**2
    imgArea = im_h * im_w
    center_w = (pred[:, 0] + pred[:, 2]) / 2
    grid_w_ind = (center_w / im_w * gwNum).astype(np.int32)
    center_h = (pred[:, 1] + pred[:, 3]) / 2
    grid_h_ind = (center_h / im_h * ghNum).astype(np.int32)
    grid_indices = grid_h_ind * gwNum + grid_w_ind
    grid_weights = grid_indices * imgArea + 1.2 * (center_w - grid_w_ind * im_w / gwNum) + (center_h - grid_h_ind * im_h / ghNum)
    grid_weights = np.array([grid_weights]).T
    sorted_pred = np.c_[pred, grid_weights]
    return sorted_pred[np.argsort(sorted_pred[:, -1])][:, :-1]

def model2annotations(model_path, img_dir_list, save_dir):
    if isinstance(img_dir_list, str):
        img_dir_list = [img_dir_list]
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    model = TextDetector(model_path=model_path, device=device)    
    imglist = []
    for img_dir in img_dir_list:
        for imgname in os.listdir(img_dir):
            if Path(imgname).suffix in IMG_EXT:
                imglist.append(osp.join(imgname))
    for imgname in tqdm(imglist):
        img = cv2.imread(osp.join(img_dir, imgname))
        imname = imgname.replace(Path(imgname).suffix, '')
        poly_save_path = osp.join(save_dir, 'line-' + imname + '.txt')
        blks, mask, polys = model(img)
        bboxes, cls, _ = blks
        polys, scores = polys
        if len(polys) != 0:
            polys = polys.reshape(-1, 8)
            np.savetxt(poly_save_path, polys, fmt='%d')
            for p in polys.reshape(len(polys), -1, 2):
                cv2.polylines(img,[p],True,(255, 0, 0), thickness=2)
        cv2.imwrite(osp.join(save_dir, imgname), img)
        
        # TextDetector._transform()


def preprocess_img(img, input_size=(1024, 1024), device='cpu', bgr2rgb=True, half=False):
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in, ratio, (dw, dh) = letterbox(img, new_shape=input_size, auto=False, stride=64)
    img_in = img_in.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img_in = np.array([np.ascontiguousarray(img_in)]).astype(np.float32) / 255
    img_in = torch.from_numpy(img_in).to(device)
    if half:
        img_in = img_in.half()
    return img_in, ratio, int(dw), int(dh)

def postprocess_mask(img: torch.Tensor, thresh=None):
    # img = img.permute(1, 2, 0)
    if thresh is not None:
        img = img > thresh
    img = img * 255
    if img.device != 'cpu':
        img = img.detach().cpu()
    img = img.numpy().astype(np.uint8)
    return img

def postprocess_yolo(det, conf_thresh, nms_thresh, resize_ratio, sort_func=None):
    det = non_max_suppression(det, conf_thresh, nms_thresh)[0]
    # bbox = det[..., 0:4]
    if det.device != 'cpu':
        det = det.detach_().cpu().numpy()
    det[..., [0, 2]] = det[..., [0, 2]] * resize_ratio[0]
    det[..., [1, 3]] = det[..., [1, 3]] * resize_ratio[1]
    if sort_func is not None:
        det = sort_func(det)

    bboxes = det[..., 0:4].astype(np.int32)
    confs = np.round(det[..., 4], 3)
    cls = det[..., 5].astype(np.int32)
    return bboxes, cls, confs

class TextDetector:
    lang_list = ['eng', 'jpn']

    def __init__(self, model_path, input_size=1024, device='cpu', half=False, nms_thresh=0.35, conf_thresh=0.4) :
        super(TextDetector, self).__init__()
        # self.blk_det, self.text_seg, self.text_det = get_base_det_models(model_path, device, half)
        self.net = TextDetBase(model_path)
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size
        self.device = device
        self.half = half
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.seg_rep = SegDetectorRepresenter(thresh=0.3)

    def _decode_blk(self, blks, im_w, im_h, dw, dh, sort=True):
        out = [out.cpu().numpy() for out in non_max_suppression(blks[0], self.conf_thresh, self.nms_thresh)]
        out = np.asarray(out)
        ratio = (im_w / (self.input_size[0] - dw), im_h / (self.input_size[1] - dh))
        bbox = out[..., 0:4]
        bbox[..., [0, 2]] = bbox[..., [0, 2]] * ratio[0]
        bbox[..., [1, 3]] = bbox[..., [1, 3]] * ratio[1]
        if sort:
            out = [grid_sort(pred, im_w, im_h) for pred in out]

        blocklist = {}
        for ii, blk in enumerate(out[0]):
            blk_dict = {}
            blk_dict['lang'] = self.lang_list[int(blk[5])]
            blk_dict['lang_cls'] = int(blk[5])
            blk_dict['xyxy'] = blk[:4].astype(np.int32).tolist()
            blk_dict['confidence'] = round(blk[4], 3)
            blocklist[str(ii)+'-'+blk_dict['lang']] = blk_dict
        return {'blocklist': OrderedDict(blocklist)}

    def __call__(self, img):
        img_in, ratio, dw, dh = preprocess_img(img, input_size=self.input_size, device=self.device, half=self.half)
        im_h, im_w = img.shape[:2]
        with torch.no_grad():
            blks, mask, lines = self.net(img_in)
        
        resize_ratio = (im_w / (self.input_size[0] - dw), im_h / (self.input_size[1] - dh))
        blks = postprocess_yolo(blks[0], 
                                self.conf_thresh, 
                                self.nms_thresh, 
                                resize_ratio, 
                                sort_func=lambda det: grid_sort(det, self.input_size[0], self.input_size[1]))
        mask = postprocess_mask(mask.squeeze_())
        mask = mask[: mask.shape[0]-dh, : mask.shape[1]-dw]
        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        boxes, scores = self.seg_rep(self.input_size, lines)
        # lines = self._inverse_transform(lines[:, 0])
        lines = postprocess_mask(lines[:, 0].squeeze_())
        boxes, scores = boxes[0], scores[0]
        if boxes.size == 0 :
            polys = []
        else :
            idx = boxes.reshape(boxes.shape[0], -1).sum(axis=1) > 0
            polys, _ = boxes[idx], scores[idx]
            polys = polys.astype(np.float64)
            # ratio = (im_w / (self.input_size[0] - dw), im_h / (self.input_size[1] - dh))
            polys[:, :, 0] *= resize_ratio[0]
            polys[:, :, 1] *= resize_ratio[1]
            polys = polys.astype(np.int32)
        return blks, mask, (polys, scores)
        # for poly in polys:
        #     cv2.polylines(img,[poly],True,(255, 0, 0), thickness=2)
        # return img, mask, lines

if __name__ == '__main__':
    device = 'cpu'
    model_path = 'data/textdetector.pt'
    textdet = TextDetector(model_path, device=device, input_size=1024)

    img_dir = r'E:\learning\wan-master\data\testpacks\jpn2'
    save_dir = r'data\dataset\result'
    model2annotations(model_path, img_dir, save_dir)
    # save_dir = osp.join(img_dir, 'rst')
    # os.makedirs(save_dir, exist_ok=True)
    # for img_path in tqdm(glob.glob(osp.join(img_dir, '*.jpg'))):
    #     img = cv2.imread(img_path)
        # img, mask, lines = textdet(img)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.imwrite(osp.join(save_dir, osp.basename(img_path)), img)
        # cv2.imwrite(osp.join(save_dir, 'mask-'+osp.basename(img_path)), mask)
        # cv2.imwrite(osp.join(save_dir, 'lines-'+osp.basename(img_path)), lines)