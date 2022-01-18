
from basemodel import TextDetBase
from utils.yolov5_utils import non_max_suppression
import os.path as osp
import os
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
from utils.imgproc_utils import letterbox, resize_keepasp, union_area, xywh2xyxypoly, xyxy2yolo, get_yololabel_strings

def grid_sort(blk_list, im_w, im_h):
    if len(blk_list) == 0:
        return blk_list
    num_ja = 0
    xyxy = []
    for blk_dict in blk_list:
        if blk_dict['language'] == 'ja':
            num_ja += 1
        xyxy.append(blk_dict['xyxy'])
    xyxy = np.array(xyxy)
    flip_lr = num_ja > len(blk_list) / 2
    im_oriw = im_w
    if im_w > im_h:
        im_w /= 2
    num_gridy, num_gridx = 4, 3
    img_area = im_h * im_w
    center_x = (xyxy[:, 0] + xyxy[:, 2]) / 2
    if flip_lr:
        if im_w != im_oriw:
            center_x = im_oriw - center_x
        else:
            center_x = im_w - center_x
    grid_x = (center_x / im_w * num_gridx).astype(np.int32)
    center_y = (xyxy[:, 1] + xyxy[:, 3]) / 2
    grid_y = (center_y / im_h * num_gridy).astype(np.int32)
    grid_indices = grid_y * num_gridx + grid_x
    grid_weights = grid_indices * img_area + 1.2 * (center_x - grid_x * im_w / num_gridx) + (center_y - grid_y * im_h / num_gridy)
    if im_w != im_oriw:
        grid_weights[np.where(grid_x >= num_gridx)] += img_area * num_gridy * num_gridx
    
    for blk_dict, weight in zip(blk_list, grid_weights):
        blk_dict['weight'] = weight
    blk_list = sorted(blk_list, key=lambda blk_dict: blk_dict['weight'])
    for blk_dict in blk_list:
        blk_dict.pop('weight')
    return blk_list

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
        for blk_dict in blk_list:
            polys += blk_dict['lines']
            blk_xyxy.append(blk_dict['xyxy'])
        blk_xyxy = xyxy2yolo(blk_xyxy, im_w, im_h)
        if blk_xyxy is not None:
            cls_list = [1] * len(blk_xyxy)
            yolo_label = get_yololabel_strings(cls_list, blk_xyxy)
        else:
            yolo_label = ''
        with open(osp.join(save_dir, imname+'.txt'), 'w', encoding='utf8') as f:
            f.write(yolo_label)

        if len(polys) != 0:
            if isinstance(polys, list):
                polys = np.array(polys)
            polys = polys.reshape(-1, 8)
            np.savetxt(poly_save_path, polys, fmt='%d')
            for p in polys.reshape(len(polys), -1, 2):
                cv2.polylines(img,[p],True,(255, 0, 0), thickness=2)
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
    for ii, blk_dict in enumerate(blk_list):
        bx1, by1, bx2, by2 = blk_dict['xyxy']
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (127, 255, 127), lw)
        lines = blk_dict['lines']
        for jj, line in enumerate(lines):
            cv2.putText(canvas, str(jj), line[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,127,0), 1)
            cv2.polylines(canvas, [line], True, (0,127,255), 2)
        center = [int((bx1 + bx2)/2), int((by1 + by2)/2)]
        cv2.putText(canvas, str(blk_dict['angle']), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (127,127,255), 2)
        # t_w, t_h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=lw)[0]
        cv2.putText(canvas, str(ii), (bx1, by1 + lw + 2), 0, lw / 3, (255,127,127), max(lw-1, 1), cv2.LINE_AA)



def sort_textlines(lines, im_w: int, im_h: int, eval_orientation: bool):
    if isinstance(lines, list):
        lines = np.array(lines, dtype=np.float64)
    middle_pnts = (lines[:, [1, 2, 3, 0]] + lines) / 2
    vec_v = middle_pnts[:, 2] - middle_pnts[:, 0]   # vertical vectors of textlines
    vec_h = middle_pnts[:, 1] - middle_pnts[:, 3]   # horizontal vectors of textlines
    # if sum of vertical vectors is longer, then text orientation is vertical, and vice versa.
    center_pnts = (lines[:, 0] + lines[:, 2]) / 2
    v = np.sum(vec_v, axis=0)
    h = np.sum(vec_h, axis=0)
    norm_v, norm_h = np.linalg.norm(v), np.linalg.norm(h)
    vertical = eval_orientation and norm_v > norm_h
    # calcuate distance between textlines and origin 
    if vertical:
        orientation_vec, orientation_norm = v, norm_v
        distance_vectors = center_pnts - np.array([[im_w, 0]], dtype=np.float64)   # vertical manga text is read from right to left, so origin is (imw, 0)
        font_size = int(round(norm_h / len(lines)))
    else:
        orientation_vec, orientation_norm = h, norm_h
        distance_vectors = center_pnts - np.array([[0, 0]], dtype=np.float64)
        font_size = int(round(norm_v / len(lines)))
    rotation_angle = int(math.atan2(orientation_vec[1], orientation_vec[0]) / math.pi * 180)     # rotation angle of textlines
    distance = np.linalg.norm(distance_vectors, axis=1)     # distance between textlinecenters and origin
    rad_matrix = np.arccos(np.einsum('ij, j->i', distance_vectors, orientation_vec) / (distance * orientation_norm))
    distance = np.abs(np.sin(rad_matrix) * distance)
    idx = np.argsort(distance)
    distance = distance[idx]
    lines = lines[idx].astype(np.int32)
    return lines, distance, rotation_angle, font_size, vertical

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


    def group_output(self, blks, lines, mask, expand_blk=True, sort_blklist=True, debug_canvas=None):
        im_h, im_w = mask.shape[:2]
        blk_list = []
        for bbox, cls, conf in zip(*blks):
            blk_dict = dict()
            blk_dict['xyxy'] = bbox
            blk_dict['lines'] = list()
            blk_dict['language'] = self.lang_list[cls]
            blk_dict['lines'] = list()
            blk_dict['vertical'] = False
            blk_list.append(blk_dict)

        # step1: filter & assign lines to textblocks
        # dbnet generate some lines inside others, maybe need to filter out those too
        bbox_score_thresh = 0.4
        mask_score_thresh = 0.1
        for ii, line in enumerate(lines):
            bx1, bx2 = line[:, 0].min(), line[:, 0].max()
            by1, by2 = line[:, 1].min(), line[:, 1].max()
            bbox_score, bbox_idx = -1, -1

            line_area = (by2-by1) * (bx2-bx1)
            for ii, blk_dict in enumerate(blk_list):
                score = union_area(blk_dict['xyxy'], [bx1, by1, bx2, by2]) / line_area
                if bbox_score < score:
                    bbox_score = score
                    bbox_idx = ii
            if bbox_score > bbox_score_thresh:
                blk_list[bbox_idx]['lines'].append(line)
            else:   # if no textblock was assigned, check whether there is "enough" textmask
                mask_score = mask[by1: by2, bx1: bx2].mean() / 255
                if mask_score > mask_score_thresh:
                    blk_dict = dict()
                    blk_dict['xyxy'] = [bx1, by1, bx2, by2]
                    blk_dict['lines'] = list()
                    blk_dict['language'] = 'unknown'
                    blk_dict['lines'] = [line]
                    blk_dict['vertical'] = False
                    blk_list.append(blk_dict)

        # step2: filter textblocks, sort textlines
        final_blk_list = list()
        for ii, blk_dict in enumerate(blk_list):
            # filter textblocks 
            if len(blk_dict['lines']) == 0:
                bx1, by1, bx2, by2 = blk_dict['xyxy']
                mask_score = mask[by1: by2, bx1: bx2].mean() / 255
                if mask_score < mask_score_thresh:
                    continue
                else:
                    xywh = np.array([[bx1, by1, bx2-bx1, by2-by1]])
                    blk_dict['lines'] = xywh2xyxypoly(xywh).reshape(-1, 4, 2).tolist()
            lines = np.array(blk_dict['lines']).astype(np.float64)
            eval_orientation = blk_dict['language'] != 'eng'
            lines, distance, blk_dict['angle'], font_size, blk_dict['vertical'] = sort_textlines(lines, im_w, im_h, eval_orientation)
            blk_dict['lines'], blk_dict['font_size'] = lines.tolist(), font_size
            # split manga text if there is a distance gap
            textblock_splitted = blk_dict['language'] == 'ja' and len(blk_dict['lines']) > 1
            if textblock_splitted:
                distance_tol = font_size * 2
                current_dict = copy.deepcopy(blk_dict)
                current_dict['lines'] = [lines[0]]
                sub_blkdict_list = [current_dict]
                for jj, line in enumerate(lines[1:]):
                    line_disance = distance[jj+1] - distance[jj]
                    split = False
                    if line_disance > distance_tol:
                        split = True
                    else:
                        if blk_dict['vertical'] and abs(abs(blk_dict['angle']) - 90) < 10:
                            if abs(lines[jj][0][1] - line[0][1]) > font_size:
                                split = True
                    if split:
                        current_dict = copy.deepcopy(current_dict)
                        current_dict['lines'] = [line]
                        sub_blkdict_list.append(current_dict)
                    else:
                        current_dict['lines'].append(line)
                if len(sub_blkdict_list) > 1:
                    for current_dict in sub_blkdict_list:
                        sub_lines = np.array(current_dict['lines'])
                        current_dict['xyxy'][0] = sub_lines[..., 0].min()
                        current_dict['xyxy'][1] = sub_lines[..., 1].min()
                        current_dict['xyxy'][2] = sub_lines[..., 0].max()
                        current_dict['xyxy'][3] = sub_lines[..., 1].max()
                else:
                    textblock_splitted = False
            else:
                sub_blkdict_list = [blk_dict]
            # modify textblock to fit its textlines
            if not textblock_splitted:
                for blk_dict in sub_blkdict_list:
                    lines = blk_dict['lines']
                    if isinstance(lines, list):
                        lines = np.array(lines, dtype=np.int32)
                    blk_dict['xyxy'][0] = min(lines[..., 0].min(), blk_dict['xyxy'][0])
                    blk_dict['xyxy'][1] = min(lines[..., 1].min(), blk_dict['xyxy'][1])
                    blk_dict['xyxy'][2] = max(lines[..., 0].max(), blk_dict['xyxy'][2])
                    blk_dict['xyxy'][3] = max(lines[..., 1].max(), blk_dict['xyxy'][3])
            final_blk_list += sub_blkdict_list

        if sort_blklist:
            final_blk_list = grid_sort(final_blk_list, im_w, im_h)
        
        if debug_canvas is not None:
            visualize_annotations(debug_canvas, final_blk_list)
            cv2.imshow('canvas', debug_canvas)
            cv2.waitKey(0)

        return final_blk_list

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

        return mask, self.group_output(blks, lines, mask, debug_canvas=None)

        # lines_map = postprocess_mask(lines_map[:, 0].squeeze_())
        # return blks, mask, lines


if __name__ == '__main__':
    device = 'cpu'
    model_path = 'data/textdetector.pt'
    # textdet = TextDetector(model_path, device=device, input_size=1024, act=True)

    # img_dir = r'D:\neonbub\mainproj\wan\data\testpacks\eng'
    img_dir = r'E:\learning\wan-master\data\testpacks\eng'
    save_dir = r'data\dataset\result'
    # model2annotations(model_path, img_dir, save_dir)
    cuda = True
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(r'data\textdetector.pt.onnx', providers=providers)
    
    # img = cv2.imread(r'E:\learning\wan-master\data\testpacks\eng\000054.jpg')
    # img_in, ratio, dw, dh = preprocess_img(img, to_tensor=False)
    # cv2.dnn.readOnnx()