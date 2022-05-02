from typing import List
import numpy as np
from shapely.geometry import Polygon
import math
import copy
from utils.imgproc_utils import union_area, xywh2xyxypoly, rotate_polygons
import cv2

LANG_LIST = ['eng', 'ja', 'unknown']
LANGCLS2IDX = {'eng': 0, 'ja': 1, 'unknown': 2}

class TextBlock(object):
    def __init__(self, xyxy: List, 
                       lines: List = None, 
                       language: str = 'unknown',
                       vertical: bool = False, 
                       font_size: float = -1,
                       distance: List = None,
                       angle: int = 0,
                       vec: List = None,
                       norm: float = -1,
                       merged: bool = False,
                       sort_weight: float = -1,
                       text: List = None,
                       translation: str = "",
                       fg_r = 0,
                       fg_g = 0,
                       fg_b = 0,
                       bg_r = 0,
                       bg_g = 0,
                       bg_b = 0,                
                       line_spacing = 1.,
                       font_family: str = "",
                       bold: bool = False,
                       underline: bool = False,
                       italic: bool = False,
                       alignment: int = -1,
                       alpha: float = 255,
                       rich_text: str = "",
                       _bounding_rect: List = None,
                       accumulate_color = True,
                       default_stroke_width = 0.2,
                       font_weight = 50, 
                       target_lang: str = "",
                       **kwargs) -> None:
        self.xyxy = [int(num) for num in xyxy]                    # boundingbox of textblock
        self.lines = [] if lines is None else lines     # polygons of textlines
        self.vertical = vertical            # orientation of textlines
        self.language = language
        self.font_size = font_size          # font pixel size
        self.distance = None if distance is None else np.array(distance, np.float64)   # distance between textlines and "origin"          
        self.angle = angle                  # rotation angle of textlines

        self.vec = None if vec is None else np.array(vec, np.float64) # primary vector of textblock
        self.norm = norm                    # primary norm of textblock
        self.merged = merged
        self.sort_weight = sort_weight

        self.text = text if text is not None else []
        self.prob = 1

        self.translation = translation

        # note they're accumulative rgb values of textlines
        self.fg_r = fg_r                       
        self.fg_g = fg_g
        self.fg_b = fg_b
        self.bg_r = bg_r
        self.bg_g = bg_g
        self.bg_b = bg_b

        # self.stroke_width = stroke_width
        self.font_family: str = font_family
        self.bold: bool = bold
        self.underline: bool = underline
        self.italic: bool = italic
        self.alpha = alpha
        self.rich_text = rich_text
        self.line_spacing = line_spacing
        # self.alignment = alignment
        self._alignment = alignment
        self._target_lang = target_lang

        self._bounding_rect = _bounding_rect
        self.default_stroke_width = default_stroke_width
        self.font_weight = font_weight
        self.accumulate_color = accumulate_color

    def adjust_bbox(self, with_bbox=False):
        lines = self.lines_array().astype(np.int32)
        if with_bbox:
            self.xyxy[0] = min(lines[..., 0].min(), self.xyxy[0])
            self.xyxy[1] = min(lines[..., 1].min(), self.xyxy[1])
            self.xyxy[2] = max(lines[..., 0].max(), self.xyxy[2])
            self.xyxy[3] = max(lines[..., 1].max(), self.xyxy[3])
        else:
            self.xyxy[0] = lines[..., 0].min()
            self.xyxy[1] = lines[..., 1].min()
            self.xyxy[2] = lines[..., 0].max()
            self.xyxy[3] = lines[..., 1].max()

    def sort_lines(self):
        if self.distance is not None:
            idx = np.argsort(self.distance)
            self.distance = self.distance[idx]
            lines = np.array(self.lines, dtype=np.int32)
            self.lines = lines[idx].tolist()

    def lines_array(self, dtype=np.float64):
        return np.array(self.lines, dtype=dtype)

    def aspect_ratio(self) -> float:
        min_rect = self.min_rect()
        middle_pnts = (min_rect[:, [1, 2, 3, 0]] + min_rect) / 2
        norm_v = np.linalg.norm(middle_pnts[:, 2] - middle_pnts[:, 0])
        norm_h = np.linalg.norm(middle_pnts[:, 1] - middle_pnts[:, 3])
        return norm_v / norm_h

    def center(self):
        xyxy = np.array(self.xyxy)
        return (xyxy[:2] + xyxy[2:]) / 2
    
    def min_rect(self, rotate_back=True):
        angled = self.angle != 0
        center = self.center()
        polygons = self.lines_array().reshape(-1, 8)
        if angled:
            polygons = rotate_polygons(center, polygons, self.angle)
        min_x = polygons[:, ::2].min()
        min_y = polygons[:, 1::2].min()
        max_x = polygons[:, ::2].max()
        max_y = polygons[:, 1::2].max()
        min_bbox = np.array([[min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]])
        if angled and rotate_back:
            min_bbox = rotate_polygons(center, min_bbox, -self.angle)
        return min_bbox.reshape(-1, 4, 2)

    # equivalent to qt's boundingRect, ignore angle
    def bounding_rect(self):
        if self._bounding_rect is None:
        # if True:
            min_bbox = self.min_rect(rotate_back=False)[0]
            x, y = min_bbox[0]
            w, h = min_bbox[2] - min_bbox[0]
            return [x, y, w, h]
        return self._bounding_rect

    def __getattribute__(self, name: str):
        if name == 'pts':
            return self.lines_array()
        # else:
        return object.__getattribute__(self, name)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]

    def to_dict(self, extra_info=False):
        blk_dict = copy.deepcopy(vars(self))
        # if not extra_info:
            # blk_dict.pop('distance')
            # blk_dict.pop('weight')
            # blk_dict.pop('vec')
            # blk_dict.pop('norm')
        return blk_dict

    def get_transformed_region(self, img, idx, textheight) -> np.ndarray :
        im_h, im_w = img.shape[:2]
        direction = 'v' if self.vertical else 'h'
        src_pts = np.array(self.lines[idx], dtype=np.float64)

        if self.language == 'eng' or (self.language == 'unknown' and not self.vertical):
            e_size = self.font_size / 3
            src_pts[..., 0] += np.array([-e_size, e_size, e_size, -e_size])
            src_pts[..., 1] += np.array([-e_size, -e_size, e_size, e_size])
            src_pts[..., 0] = np.clip(src_pts[..., 0], 0, im_w)
            src_pts[..., 1] = np.clip(src_pts[..., 1], 0, im_h)

        middle_pnt = (src_pts[[1, 2, 3, 0]] + src_pts) / 2
        vec_v = middle_pnt[2] - middle_pnt[0]   # vertical vectors of textlines
        vec_h = middle_pnt[1] - middle_pnt[3]   # horizontal vectors of textlines
        ratio = np.linalg.norm(vec_v) / np.linalg.norm(vec_h)

        if direction == 'h' :
            h = int(textheight)
            w = int(round(textheight / ratio))
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            region = cv2.warpPerspective(img, M, (w, h))
        elif direction == 'v' :
            w = int(textheight)
            h = int(round(textheight * ratio))
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            region = cv2.warpPerspective(img, M, (w, h))
            region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # cv2.imshow('region'+str(idx), region)
        # cv2.waitKey(0)
        return region

    def get_text(self):
        if isinstance(self.text, str):
            return self.text
        return ' '.join(self.text).strip()

    def set_font_colors(self, frgb, srgb, accumulate=True):
        self.accumulate_color = accumulate
        num_lines = len(self.lines) if accumulate and len(self.lines) > 0 else 1
        # set font color
        frgb = np.array(frgb) * num_lines
        self.fg_r, self.fg_g, self.fg_b = frgb
        # set stroke color  
        srgb = np.array(srgb) * num_lines
        self.bg_r, self.bg_g, self.bg_b = srgb

    def get_font_colors(self, bgr=False):
        num_lines = len(self.lines)
        frgb = np.array([self.fg_r, self.fg_g, self.fg_b])
        brgb = np.array([self.bg_r, self.bg_g, self.bg_b])
        if self.accumulate_color:
            if num_lines > 0:
                frgb = (frgb / num_lines).astype(np.int32)
                brgb = (brgb / num_lines).astype(np.int32)
                if bgr:
                    return frgb[::-1], brgb[::-1]
                else:
                    return frgb, brgb
            else:
                return [0, 0, 0], [0, 0, 0]
        else:
            return frgb, brgb

    def xywh(self):
        x, y, w, h = self.xyxy
        return [x, y, w-x, h-y]

    # alignleft: 0, center: 1, right: 2 
    def alignment(self):
        if self._alignment >= 0:
            return self._alignment
        elif self.vertical:
            return 0
        lines = self.lines_array()
        if len(lines) == 1:
            return 0
        angled = self.angle != 0
        polygons = lines.reshape(-1, 8)
        if angled:
            polygons = rotate_polygons((0, 0), polygons, self.angle)
        polygons = polygons.reshape(-1, 4, 2)
        
        left_std = np.std(polygons[:, 0, 0])
        # right_std = np.std(polygons[:, 1, 0])
        center_std = np.std((polygons[:, 0, 0] + polygons[:, 1, 0]) / 2)
        if left_std < center_std:
            return 0
        else:
            return 1

    def target_lang(self):
        return self.target_lang

    @property
    def stroke_width(self):
        var = np.array([self.fg_r, self.fg_g, self.fg_b]) \
            - np.array([self.bg_r, self.bg_g, self.bg_b])
        var = np.abs(var).sum()
        if var > 40:
            return self.default_stroke_width
        return 0

def sort_textblk_list(blk_list: List[TextBlock], im_w: int, im_h: int) -> List[TextBlock]:
    if len(blk_list) == 0:
        return blk_list
    num_ja = 0
    xyxy = []
    for blk in blk_list:
        if blk.language == 'ja':
            num_ja += 1
        xyxy.append(blk.xyxy)
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
    
    for blk, weight in zip(blk_list, grid_weights):
        blk.sort_weight = weight
    blk_list.sort(key=lambda blk: blk.sort_weight)
    return blk_list

def examine_textblk(blk: TextBlock, im_w: int, im_h: int, eval_orientation: bool, sort: bool = False) -> None:
    lines = blk.lines_array()
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
        primary_vec, primary_norm = v, norm_v
        distance_vectors = center_pnts - np.array([[im_w, 0]], dtype=np.float64)   # vertical manga text is read from right to left, so origin is (imw, 0)
        font_size = int(round(norm_h / len(lines)))
    else:
        primary_vec, primary_norm = h, norm_h
        distance_vectors = center_pnts - np.array([[0, 0]], dtype=np.float64)
        font_size = int(round(norm_v / len(lines)))
    
    rotation_angle = int(math.atan2(primary_vec[1], primary_vec[0]) / math.pi * 180)     # rotation angle of textlines
    distance = np.linalg.norm(distance_vectors, axis=1)     # distance between textlinecenters and origin
    rad_matrix = np.arccos(np.einsum('ij, j->i', distance_vectors, primary_vec) / (distance * primary_norm))
    distance = np.abs(np.sin(rad_matrix) * distance)
    blk.lines = lines.astype(np.int32).tolist()
    blk.distance = distance
    blk.angle = rotation_angle
    if vertical:
        blk.angle -= 90
    blk.font_size = font_size
    if eval_orientation:
        blk.vertical = vertical
    blk.vec = primary_vec
    blk.norm = primary_norm
    if sort:
        blk.sort_lines()

def try_merge_textline(blk: TextBlock, blk2: TextBlock, fntsize_tol=1.3, distance_tol=2) -> bool:
    if blk2.merged:
        return False
    fntsize_div = blk.font_size / blk2.font_size
    num_l1, num_l2 = len(blk), len(blk2)
    fntsz_avg = (blk.font_size * num_l1 + blk2.font_size * num_l2) / (num_l1 + num_l2)
    vec_prod = blk.vec @ blk2.vec
    vec_sum = blk.vec + blk2.vec
    cos_vec = vec_prod / blk.norm / blk2.norm
    distance = blk2.distance[-1] - blk.distance[-1]
    distance_p1 = np.linalg.norm(np.array(blk2.lines[-1][0]) - np.array(blk.lines[-1][0]))
    l1, l2 = Polygon(blk.lines[-1]), Polygon(blk2.lines[-1])
    if not l1.intersects(l2):
        if fntsize_div > fntsize_tol or 1 / fntsize_div > fntsize_tol:
            return False
        if abs(cos_vec) < 0.866:   # cos30
            return False
        if distance > distance_tol * fntsz_avg or distance_p1 > fntsz_avg * 2.5:
            return False
    # merge
    blk.lines.append(blk2.lines[0])
    blk.vec = vec_sum
    blk.angle = int(round(np.rad2deg(math.atan2(vec_sum[1], vec_sum[0]))))
    if blk.vertical:
        blk.angle -= 90
    blk.norm = np.linalg.norm(vec_sum)
    blk.distance = np.append(blk.distance, blk2.distance[-1])
    blk.font_size = fntsz_avg
    blk2.merged = True
    return True

def merge_textlines(blk_list: List[TextBlock]) -> List[TextBlock]:
    if len(blk_list) < 2:
        return blk_list
    blk_list.sort(key=lambda blk: blk.distance[0])
    merged_list = []
    for ii, current_blk in enumerate(blk_list):
        if current_blk.merged:
            continue
        for jj, blk in enumerate(blk_list[ii+1:]):
            try_merge_textline(current_blk, blk)
        merged_list.append(current_blk)
    for blk in merged_list:
        blk.adjust_bbox(with_bbox=False)
    return merged_list

def split_textblk(blk: TextBlock):
    font_size, distance, lines = blk.font_size, blk.distance, blk.lines_array()
    distance_tol = font_size * 2
    current_blk = copy.deepcopy(blk)
    current_blk.lines = [lines[0]]
    sub_blk_list = [current_blk]
    textblock_splitted = False
    for jj, line in enumerate(lines[1:]):
        l1, l2 = Polygon(lines[jj]), Polygon(line)
        split = False
        if not l1.intersects(l2):
            line_disance = distance[jj+1] - distance[jj]
            if line_disance > distance_tol:
                split = True
            else:
                if blk.vertical and abs(abs(blk.angle) - 90) < 10:
                    split = abs(lines[jj][0][1] - line[0][1]) > font_size
        if split:
            current_blk = copy.deepcopy(current_blk)
            current_blk.lines = [line]
            sub_blk_list.append(current_blk)
        else:
            current_blk.lines.append(line)
    if len(sub_blk_list) > 1:
        textblock_splitted = True
        for current_blk in sub_blk_list:
            current_blk.adjust_bbox(with_bbox=False)
    return textblock_splitted, sub_blk_list

def group_output(blks, lines, im_w, im_h, mask=None, sort_blklist=True) -> List[TextBlock]:
    blk_list, scattered_lines = [], {'ver': [], 'hor': []}
    for bbox, cls, conf in zip(*blks):
        blk_list.append(TextBlock(bbox, language=LANG_LIST[cls]))

    # step1: filter & assign lines to textblocks
    bbox_score_thresh = 0.4
    mask_score_thresh = 0.1
    for ii, line in enumerate(lines):
        bx1, bx2 = line[:, 0].min(), line[:, 0].max()
        by1, by2 = line[:, 1].min(), line[:, 1].max()
        bbox_score, bbox_idx = -1, -1
        line_area = (by2-by1) * (bx2-bx1)
        for jj, blk in enumerate(blk_list):
            score = union_area(blk.xyxy, [bx1, by1, bx2, by2]) / line_area
            if bbox_score < score:
                bbox_score = score
                bbox_idx = jj
        if bbox_score > bbox_score_thresh:
            blk_list[bbox_idx].lines.append(line)
        else:   # if no textblock was assigned, check whether there is "enough" textmask
            if mask is not None:
                mask_score = mask[by1: by2, bx1: bx2].mean() / 255
                if mask_score < mask_score_thresh:
                    continue
            blk = TextBlock([bx1, by1, bx2, by2], [line])
            examine_textblk(blk, im_w, im_h, True, sort=False)
            if blk.vertical:
                scattered_lines['ver'].append(blk)
            else:
                scattered_lines['hor'].append(blk)

    # step2: filter textblocks, sort & split textlines
    final_blk_list = []
    for ii, blk in enumerate(blk_list):
        # filter textblocks 
        if len(blk.lines) == 0:
            bx1, by1, bx2, by2 = blk.xyxy
            if mask is not None:
                mask_score = mask[by1: by2, bx1: bx2].mean() / 255
                if mask_score < mask_score_thresh:
                    continue
            xywh = np.array([[bx1, by1, bx2-bx1, by2-by1]])
            blk.lines = xywh2xyxypoly(xywh).reshape(-1, 4, 2).tolist()
        lines = blk.lines_array()
        eval_orientation = blk.language != 'eng'
        examine_textblk(blk, im_w, im_h, eval_orientation, sort=True)
        
        # split manga text if there is a distance gap
        textblock_splitted = blk.language == 'ja' and len(blk.lines) > 1
        if textblock_splitted:
            textblock_splitted, sub_blk_list = split_textblk(blk)
        else:
            sub_blk_list = [blk]
        # modify textblock to fit its textlines
        if not textblock_splitted:
            for blk in sub_blk_list:
                blk.adjust_bbox(with_bbox=True)
        final_blk_list += sub_blk_list

    # step3: merge scattered lines, sort textblocks by "grid"
    final_blk_list += merge_textlines(scattered_lines['hor'])
    final_blk_list += merge_textlines(scattered_lines['ver'])
    if sort_blklist:
        final_blk_list = sort_textblk_list(final_blk_list, im_w, im_h)
    return final_blk_list

def visualize_textblocks(canvas, blk_list:  List[TextBlock]):
    lw = max(round(sum(canvas.shape) / 2 * 0.003), 2)  # line width
    for ii, blk in enumerate(blk_list):
        bx1, by1, bx2, by2 = blk.xyxy
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (127, 255, 127), lw)
        lines = blk.lines_array(dtype=np.int32)
        for jj, line in enumerate(lines):
            cv2.putText(canvas, str(jj), line[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,127,0), 1)
            cv2.polylines(canvas, [line], True, (0,127,255), 2)
        cv2.polylines(canvas, [blk.min_rect()], True, (127,127,0), 2)
        center = [int((bx1 + bx2)/2), int((by1 + by2)/2)]
        cv2.putText(canvas, str(blk.angle), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (127,127,255), 2)
        cv2.putText(canvas, str(ii), (bx1, by1 + lw + 2), 0, lw / 3, (255,127,127), max(lw-1, 1), cv2.LINE_AA)
    return canvas