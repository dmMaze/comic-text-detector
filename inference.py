
from typing import List
from basemodel import TextDetBase
from utils.yolov5_utils import non_max_suppression
import os.path as osp
from tqdm import tqdm
import numpy as np
import cv2
import torch
from pathlib import Path
import torch
import onnxruntime
import math
import copy
from utils.db_utils import SegDetectorRepresenter
from utils.imgio_utils import imread, imwrite, find_all_imgs
from utils.imgproc_utils import letterbox, xyxy2yolo, get_yololabel_strings
from utils.textblock import TextBlock, group_output
from shapely.geometry import Polygon
import random

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LANG_ENG = 0
LANG_JPN = 1

def expand_textwindow(img_size, xyxy, expand_r=8, shrink=False):
    im_h, im_w = img_size[:2]
    x1, y1 , x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    paddings = int(round((max(h, w) * 0.25 + min(h, w) * 0.75) / expand_r))
    if shrink:
        paddings *= -1
    x1, y1 = max(0, x1 - paddings), max(0, y1 - paddings)
    x2, y2 = min(im_w-1, x2+paddings), min(im_h-1, y2+paddings)
    return [x1, y1, x2, y2]

def extract_textballoon(img, pred_textmsk=None, global_mask=None):
    if len(img.shape) > 2 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im_h, im_w = img.shape[0], img.shape[1]
    hyp_textmsk = np.zeros((im_h, im_w), np.uint8)

    thresh_val, threshed = cv2.threshold(img, 1, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    
    xormap_sum = cv2.bitwise_xor(threshed, pred_textmsk).sum()
    neg_threshed = 255 - threshed
    neg_xormap_sum = cv2.bitwise_xor(neg_threshed, pred_textmsk).sum()
    neg_thresh = neg_xormap_sum < xormap_sum
    if neg_thresh:
        threshed = neg_threshed
    thresh_info = {'thresh_val': thresh_val,'neg_thresh': neg_thresh}
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(threshed, connectivity, cv2.CV_16U)
    label_unchanged = np.copy(labels)
    if global_mask is not None:
        labels[np.where(global_mask==0)] = 0
    text_labels = []
    if pred_textmsk is not None:
        text_score_thresh = 0.5
        textbbox_map = np.zeros_like(pred_textmsk)
        for label_index, stat, centroid in zip(range(num_labels), stats, centroids):
            if label_index != 0: # skip background label
                x, y, w, h, area = stat
                area *= 255
                x1, y1, x2, y2 = x, y, x+w, y+h
                label_local = labels[y1: y2, x1: x2]
                label_cordinates = np.where(label_local==label_index)
                tmp_text_map = np.zeros((h, w), np.uint8)
                tmp_text_map[label_cordinates] = 255
                andmap = cv2.bitwise_and(tmp_text_map, pred_textmsk[y1: y2, x1: x2])
                text_score = andmap.sum() / area
                if text_score > text_score_thresh:
                    text_labels.append(label_index)
                    hyp_textmsk[y1: y2, x1: x2][label_cordinates] = 255
    labels = label_unchanged
    bubble_msk = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    bubble_msk[np.where(labels==0)] = 255
    # if lang == LANG_JPN:
    bubble_msk = cv2.erode(bubble_msk, (3, 3), iterations=1)
    line_thickness = 2
    cv2.rectangle(bubble_msk, (0, 0), (im_w, im_h), BLACK, line_thickness, cv2.LINE_8)
    contours, hiers = cv2.findContours(bubble_msk, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    brect_area_thresh = im_h * im_w * 0.4
    min_brect_area = np.inf
    ballon_index = -1
    maxium_pixsum = -1
    for ii, contour in enumerate(contours):
        brect = cv2.boundingRect(contours[ii])
        brect_area = brect[2] * brect[3]
        if brect_area > brect_area_thresh and brect_area < min_brect_area:
            tmp_ballonmsk = np.zeros_like(bubble_msk)
            tmp_ballonmsk = cv2.drawContours(tmp_ballonmsk, contours, ii, WHITE, cv2.FILLED)
            andmap_sum = cv2.bitwise_and(tmp_ballonmsk, hyp_textmsk).sum()
            if andmap_sum > maxium_pixsum:
                maxium_pixsum = andmap_sum
                min_brect_area = brect_area
                ballon_index = ii
    if ballon_index != -1:
        bubble_msk = np.zeros_like(bubble_msk)
        bubble_msk = cv2.drawContours(bubble_msk, contours, ballon_index, WHITE, cv2.FILLED)
    hyp_textmsk = cv2.bitwise_and(hyp_textmsk, bubble_msk)
    return hyp_textmsk, bubble_msk, thresh_info, (num_labels, label_unchanged, stats, centroids, text_labels)

def extract_textballoon_channelwise(img, pred_textmsk, test_grey=True, global_mask=None):
    c_list = [img[:, :, i] for i in range(3)]
    if test_grey:
        c_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    best_xorpix_sum = np.inf
    best_cindex = best_hyptextmsk = best_bubblemsk = best_thresh_info = best_component_stats = None
    for c_index, channel in enumerate(c_list):
        hyp_textmsk, bubble_msk, thresh_info, component_stats = extract_textballoon(channel, pred_textmsk, global_mask=global_mask)
        pixor_sum = cv2.bitwise_xor(hyp_textmsk, pred_textmsk).sum()
        if pixor_sum < best_xorpix_sum:
            best_xorpix_sum = pixor_sum
            best_cindex = c_index
            best_hyptextmsk, best_bubblemsk, best_thresh_info, best_component_stats = hyp_textmsk, bubble_msk, thresh_info, component_stats
    return best_hyptextmsk, best_bubblemsk, best_component_stats

def refine_textmask(img, pred_mask, channel_wise=True, find_leaveouts=True, global_mask=None):
    hyp_textmsk, bubble_msk, component_stats = extract_textballoon_channelwise(img, pred_mask, global_mask=global_mask)
    num_labels, labels, stats, centroids, text_labels = component_stats
    stats = np.array(stats)
    text_stats = stats[text_labels]
    if find_leaveouts and len(text_stats) > 0:
        median_h = np.median(text_stats[:, 3])
        for label, label_h in zip(range(num_labels), stats[:, 3]):
            if label == 0 or label in text_labels:
                continue
            if label_h > 0.5 * median_h and label_h < 1.5 * median_h:
                hyp_textmsk[np.where(labels==label)] = 255
        hyp_textmsk = cv2.bitwise_and(hyp_textmsk, bubble_msk)
        if global_mask is not None:
            hyp_textmsk = cv2.bitwise_and(hyp_textmsk, global_mask)
    return hyp_textmsk, bubble_msk


def draw_connected_labels(num_labels, labels, stats, centroids):

    labdraw = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    max_ind = np.argmax(stats[:, 4])
    for ind, lab in enumerate((range(num_labels))):
        if ind != max_ind:
            randcolor = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            labdraw[np.where(labels==lab)] = randcolor
            maxr, minr = 0.5, 0.001
            maxw, maxh = stats[max_ind][2] * maxr, stats[max_ind][3] * maxr
            minarea = labdraw.shape[0] * labdraw.shape[1] * minr

            stat = stats[ind]
            bboxarea = stat[2] * stat[3]
            if stat[2] < maxw and stat[3] < maxh and bboxarea > minarea:
                pix = np.zeros((labels.shape[0], labels.shape[1]), dtype=np.uint8)
                pix[np.where(labels==lab)] = 255

                rect = cv2.minAreaRect(cv2.findNonZero(pix))
                box = np.int0(cv2.boxPoints(rect))
                labdraw = cv2.drawContours(labdraw, [box], 0, randcolor, 2)
                labdraw = cv2.circle(labdraw, (int(centroids[ind][0]),int(centroids[ind][1])), radius=5, color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)), thickness=-1)                

    cv2.imshow("draw_connected_labels", labdraw)
    return labdraw




def model2annotations(model_path, img_dir_list, save_dir):
    if isinstance(img_dir_list, str):
        img_dir_list = [img_dir_list]
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    model = TextDetector(model_path=model_path, device=device, act='leaky')  
    imglist = []
    for img_dir in img_dir_list:
        imglist += find_all_imgs(img_dir, abs_path=True)
    for img_path in tqdm(imglist):
        imgname = osp.basename(img_path)
        img = cv2.imread(img_path)
        im_h, im_w = img.shape[:2]
        imname = imgname.replace(Path(imgname).suffix, '')
        maskname = 'mask-'+imname+'.png'
        poly_save_path = osp.join(save_dir, 'line-' + imname + '.txt')
        mask, blk_list = model(img)
        polys = []
        blk_xyxy = []
        for blk in blk_list:
            polys += blk.lines
            blk_xyxy.append(blk.xyxy)
        blk_xyxy = xyxy2yolo(blk_xyxy, im_w, im_h)
        if blk_xyxy is not None:
            cls_list = [1] * len(blk_xyxy)
            yolo_label = get_yololabel_strings(cls_list, blk_xyxy)
        else:
            yolo_label = ''
        with open(osp.join(save_dir, imname+'.txt'), 'w', encoding='utf8') as f:
            f.write(yolo_label)

        
        # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        # _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        # draw_connected_labels(num_labels, labels, stats, centroids)
        for blk in blk_list:
            bx1, by1, bx2, by2 = expand_textwindow(img.shape, blk.xyxy)
            # bx1, by1, bx2, by2 = blk.xyxy
            
            # hyp_textmsk, bubble_msk = refine_textmask(img[by1: by2, bx1: bx2], mask[by1: by2, bx1: bx2], find_leaveouts=True)
            im = np.ascontiguousarray(img[by1: by2, bx1: bx2])
            msk = np.ascontiguousarray(mask[by1: by2, bx1: bx2])
            # crf_mask = refine_mask(im, msk)
            # cv2.imshow('hyptext', hyp_textmsk)
            # cv2.imshow('bubble', bubble_msk)
        visualize_annotations(img, blk_list)
        cv2.imshow('rst', img)
        cv2.imshow('mask', mask)
        cv2.waitKey(0)

        if len(polys) != 0:
            if isinstance(polys, list):
                polys = np.array(polys)
            polys = polys.reshape(-1, 8)
            np.savetxt(poly_save_path, polys, fmt='%d')
            # for p in polys.reshape(len(polys), -1, 2):
            #     cv2.polylines(img,[p],True,(255, 0, 0), thickness=2)
        cv2.imwrite(osp.join(save_dir, imgname), img)
        cv2.imwrite(osp.join(save_dir, maskname), mask)

def preprocess_img(img, input_size=(1024, 1024), device='cpu', bgr2rgb=True, half=False, to_tensor=True):
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in, ratio, (dw, dh) = letterbox(img, new_shape=input_size, auto=False, stride=64)
    img_in = img_in.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img_in = np.array([np.ascontiguousarray(img_in)]).astype(np.float32) / 255
    if to_tensor:
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

    blines = det[..., 0:4].astype(np.int32)
    confs = np.round(det[..., 4], 3)
    cls = det[..., 5].astype(np.int32)
    return blines, cls, confs

def draw_textlines(img, polys,  color=(255, 0, 0)):
    for poly in polys:
        cv2.polylines(img,[poly],True,color=color, thickness=2)

def visualize_annotations(canvas, blk_list):
    lw = max(round(sum(canvas.shape) / 2 * 0.003), 2)  # line width
    for ii, blk in enumerate(blk_list):
        bx1, by1, bx2, by2 = blk.xyxy
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (127, 255, 127), lw)
        lines = blk.lines_array(dtype=np.int32)
        for jj, line in enumerate(lines):
            cv2.putText(canvas, str(jj), line[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,127,0), 1)
            cv2.polylines(canvas, [line], True, (0,127,255), 2)
        center = [int((bx1 + bx2)/2), int((by1 + by2)/2)]
        cv2.putText(canvas, str(blk.angle), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (127,127,255), 2)
        cv2.putText(canvas, str(ii), (bx1, by1 + lw + 2), 0, lw / 3, (255,127,127), max(lw-1, 1), cv2.LINE_AA)



def hex2bgr(hex):
    gmask = 254 << 8
    rmask = 254
    b = hex >> 16
    g = (hex & gmask) >> 8
    r = hex & rmask
    return np.stack([b, g, r]).transpose()

def get_topk_color(color_list, bins, k=3, color_var=20, bin_tol=0.001):
    top_colors = [color_list[0]]
    bin_tol = np.sum(bins) * bin_tol
    if len(color_list) > 1:
        for color, bin in zip(color_list[1:], bins[1:]):
            if np.abs(np.array(top_colors) - color).min() > color_var:
                top_colors.append(color)
            if len(top_colors) > k or bin < bin_tol:
                break
    return top_colors

def refine_mask(img: np.asanyarray, mask: np.asanyarray, blk_list: List[TextBlock]) -> np.asanyarray:
    for blk in blk_list:
        bx1, by1, bx2, by2 = expand_textwindow(img.shape, blk.xyxy, expand_r=16)
        # bx1, by1, bx2, by2 = blk.xyxy
        
        hyp_textmsk, bubble_msk = refine_textmask(img[by1: by2, bx1: bx2], mask[by1: by2, bx1: bx2], find_leaveouts=True)
        im = np.ascontiguousarray(img[by1: by2, bx1: bx2])
        
        im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        msk = np.ascontiguousarray(mask[by1: by2, bx1: bx2])
        # candidate_px = im[np.where(msk > 200)].astype(np.uint32)
        # candidate_px = (candidate_px[:, 0] << 16) + (candidate_px[:, 1] << 8) + candidate_px[:, 2]
        candidate_grey_px = im_grey[np.where(msk > 127)]
        bin, his = np.histogram(candidate_grey_px, bins=255)
        idx = np.argsort(bin * -1)
        his = his[idx]
        bin = bin[idx]
        # hex = hex.astype(np.uint32)
        # idx = np.argsort(bin * -1)
        # hex = hex[idx]
        # bgr = hex2bgr(hex)
        topk_color = get_topk_color(his, bin)
        color_range = 30
        for ii, color in enumerate(topk_color):
            c = np.ones((100, 100, 3)) * color
            c = c.astype(np.uint8)
            threshed = cv2.inRange(im_grey, int(color-color_range), int(color+color_range))
            # cv2.imshow(str(ii), c)
            cv2.imshow('threshed-'+str(ii), threshed)

        cv2.imshow('im', im)
        cv2.imshow('msk', msk)
        cv2.waitKey(0)
    return mask

class TextDetector:
    lang_list = ['eng', 'ja', 'unknown']
    langcls2idx = {'eng': 0, 'ja': 1, 'unknown': 2}

    def __init__(self, model_path, input_size=1024, device='cpu', half=False, nms_thresh=0.35, conf_thresh=0.4, mask_thresh=0.3, act='leaky', backend='torch') :
        super(TextDetector, self).__init__()
        cuda = device == 'cuda'
        self.backend = backend
        if self.backend == 'torch':
            self.net = TextDetBase(model_path, device=device, act=act)
        else:
            # TODO: OPENCV ONNX INF
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size
        self.device = device
        self.half = half
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.seg_rep = SegDetectorRepresenter(thresh=0.3)

    def __call__(self, img):
        img_in, ratio, dw, dh = preprocess_img(img, input_size=self.input_size, device=self.device, half=self.half)

        im_h, im_w = img.shape[:2]
        with torch.no_grad():
            blks, mask, lines_map = self.net(img_in)
        
        resize_ratio = (im_w / (self.input_size[0] - dw), im_h / (self.input_size[1] - dh))
        blks = postprocess_yolo(blks[0], 
                                self.conf_thresh, 
                                self.nms_thresh, 
                                resize_ratio, 
                                sort_func=None)
        mask = postprocess_mask(mask.squeeze_())
        lines, scores = self.seg_rep(self.input_size, lines_map)
        box_thresh = 0.6
        idx = np.where(scores[0] > box_thresh)
        lines, scores = lines[0][idx], scores[0][idx]

        # map output to input img
        mask = mask[: mask.shape[0]-dh, : mask.shape[1]-dw]
        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        if lines.size == 0 :
            lines = []
        else :
            lines = lines.astype(np.float64)
            lines[..., 0] *= resize_ratio[0]
            lines[..., 1] *= resize_ratio[1]
            lines = lines.astype(np.int32)
        blk_list = group_output(blks, lines, im_w, im_h, mask)
        mask = refine_mask(img, mask, blk_list)
        return mask, blk_list


if __name__ == '__main__':
    device = 'cpu'
    model_path = 'data/textdetector.pt'
    # textdet = TextDetector(model_path, device=device, input_size=1024, act=True)

    img_dir = r'D:\neonbub\mainproj\wan\data\testpacks\tmp'
    # img_dir = r'E:\learning\wan-master\data\testpacks\eng'
    save_dir = r'data\backup'
    model2annotations(model_path, img_dir, save_dir)
    cuda = True
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(r'data\textdetector.pt.onnx', providers=providers)
    
    # img = cv2.imread(r'E:\learning\wan-master\data\testpacks\eng\000054.jpg')
    # img_in, ratio, dw, dh = preprocess_img(img, to_tensor=False)
    # cv2.dnn.readOnnx()