
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
# import torch.nn as nn
import glob



def grid_sort(pred, im_h, im_w):
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

    def _transform(self, img, bgr2rgb=True):
        if bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in, ratio, (dw, dh) = letterbox(img, new_shape=self.input_size, auto=False, stride=64)
        img_in = img_in.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_in = np.array([np.ascontiguousarray(img_in)]).astype(np.float32) / 255
        img_in = torch.from_numpy(img_in).to(self.device)
        if self.half:
            img_in = img_in.half()
        return img_in, ratio, int(dw), int(dh)

    def _inverse_transform(self, img: torch.Tensor, thresh=None):
        img = img.permute(1, 2, 0)
        if thresh is not None:
            img = img > thresh
        img = img * 255
        img = img.cpu().numpy().astype(np.uint8)
        return img

    def _decode_blk(self, blks, im_w, im_h, dw, dh, sort=True):
        out = [out.cpu().numpy() for out in non_max_suppression(blks[0], self.conf_thresh, self.nms_thresh)]
        out = np.asarray(out)
        ratio = (im_w / (self.input_size[0] - dw), im_h / (self.input_size[1] - dh))
        bbox = out[..., 0:4]
        bbox[..., [0, 2]] = bbox[..., [0, 2]] * ratio[0]
        bbox[..., [1, 3]] = bbox[..., [1, 3]] * ratio[1]
        if sort:
            out = [grid_sort(pred, im_h, im_w) for pred in out]

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
        img_in, ratio, dw, dh = self._transform(img)
        im_h, im_w = img.shape[:2]
        with torch.no_grad():
            blks, mask, lines = self.net(img_in)
        
        textdict = self._decode_blk(blks, im_w, im_h, dw, dh)
        # for blkname, blkcontent in textdict['blocklist'].items():
        #     x1, y1, x2, y2 = blkcontent['xyxy']
        #     cv2.imshow('img', img[y1: y2, x1: x2])
        #     cv2.waitKey(0)
        mask = self._inverse_transform(mask[0])
        mask = mask[: mask.shape[0]-dh, : mask.shape[1]-dw]
        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        boxes, scores = self.seg_rep(self.input_size, lines)
        lines = self._inverse_transform(lines[:, 0])
        boxes, scores = boxes[0], scores[0]
        if boxes.size == 0 :
            polys = []
        else :
            idx = boxes.reshape(boxes.shape[0], -1).sum(axis=1) > 0
            polys, _ = boxes[idx], scores[idx]
            polys = polys.astype(np.float64)
            # polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net = 1)
            ratio = (im_w / (self.input_size[0] - dw), im_h / (self.input_size[1] - dh))
            polys[:, :, 0] *= ratio[0]
            polys[:, :, 1] *= ratio[1]
            polys = polys.astype(np.int32)
        for poly in polys:
            cv2.polylines(img,[poly],True,(255, 0, 0), thickness=2)
        return img, mask, lines

if __name__ == '__main__':
    device = 'cpu'
    model_path = 'data/textdetector.pt'
    textdet = TextDetector(model_path, device=device, input_size=1024)

    img_dir = r'E:\learning\wan-master\data\testpacks\jpn2'

    save_dir = osp.join(img_dir, 'rst')
    os.makedirs(save_dir, exist_ok=True)
    for img_path in tqdm(glob.glob(osp.join(img_dir, '*.jpg'))):
        img = cv2.imread(img_path)
        img, mask, lines = textdet(img)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        # cv2.imwrite(osp.join(save_dir, osp.basename(img_path)), img)
        # cv2.imwrite(osp.join(save_dir, 'mask-'+osp.basename(img_path)), mask)
        # cv2.imwrite(osp.join(save_dir, 'lines-'+osp.basename(img_path)), lines)