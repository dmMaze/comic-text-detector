from copy import copy
from http.client import IM_USED
import pathlib
import shutil
import PIL
import cv2

import numpy as np
import os.path as osp
import os
from PIL import Image, ImageColor, ImageFont, ImageDraw, ImageFilter, ImageOps
import random

from numpy.random import rand
from trdg.utils import load_dict, load_fonts
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append(os.getcwd())
from utils.io_utils import find_all_imgs, imread, imwrite
from utils.imgproc_utils import *
import copy

ALIGN_LEFT = 0
ALIGN_CENTER = 1
ALIGN_RIGHT = 2

ORIENTATION_HOR = 0
ORIENTATION_VER = 1

def get_textlines_from_langdict(lang_dict, num_line, line_len, sampler=None):
    textlines = []
    dict_len = len(lang_dict)
    for ii in range(num_line):
        line = ''
        for jj in range(line_len):
            line += lang_dict[random.randrange(dict_len)] + ' '
        textlines.append(line[:line_len])
    if sampler is None:
        return textlines
    return textlines

def draw_text_polygons(img, text_polygons, color=None):
    if isinstance(img, PIL.Image.Image):
        img = np.array(img)
    img = np.copy(img)
    for poly in text_polygons:
        if color is None:
            randcolor = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        else:
            randcolor = color
        cv2.polylines(img,[poly.reshape((-1, 1, 2))],True,randcolor, thickness=2)
    return img

def draw_textblk(textlines, font, 
                     fill='black',
                     stroke_width=0,
                     stroke_fill='grey',
                     spacing=0,
                     rotation=0,
                     orientation=ORIENTATION_HOR,
                     alignment=ALIGN_LEFT):

    text_size = np.array([font.getsize(line) for line in textlines])
    if orientation == ORIENTATION_HOR:
        line_widths, line_heights = text_size[:, 0], text_size[:, 1]
        textblk_w = max(text_size[:, 0]) + 3*stroke_width 
        textblk_h = (len(textlines) - 1) * spacing + text_size[:, 1].sum() + 3*stroke_width
    else:
        line_widths, line_heights = text_size[:, 1], text_size[:, 0]
        textblk_w = line_widths.sum() + 3*stroke_width 
        textblk_h = max(line_heights) + 3*stroke_width
    if orientation == ORIENTATION_VER:
        textblk_h += font.size * 3  # some fonts are not correctly aligned
    
    txtblk_img = Image.new("RGBA", (textblk_w, textblk_h), (255, 255, 255, 255))
    txtblk_draw = ImageDraw.Draw(txtblk_img)
    txtblk_draw.fontmode = '1'      # disable anti-aliasing
    txtblk_mask = Image.new("L", (textblk_w, textblk_h), (0))
    tmp_msk = txtblk_mask.copy()
    tmp_msk_draw = ImageDraw.Draw(tmp_msk)
    tmp_msk_draw.fontmode = '1'

    textpolygons = []
    if orientation == ORIENTATION_VER:     
        for ii, line in enumerate(textlines):
            x_offset = sum(line_widths[:ii]) + stroke_width
            for jj, char in enumerate(line):
                txtblk_draw.text((x_offset, jj*font.size), char, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
                tmp_msk_draw.text((x_offset, jj*font.size), char, font=font, fill='white', stroke_width=stroke_width, stroke_fill='white')
            valid_bbox = tmp_msk.getbbox()
            if valid_bbox is None:
                continue
            txtblk_mask.paste(tmp_msk, mask=tmp_msk)
            tmp_msk.paste('black', [0, 0, tmp_msk.size[0],tmp_msk.size[1]])
            textpolygons.append([valid_bbox[0], valid_bbox[1], valid_bbox[2]-valid_bbox[0], valid_bbox[3]-valid_bbox[1]])
    else:
        for ii, line in enumerate(textlines):
            x_offset = stroke_width
            y_offset = sum(line_heights[0:ii]) + stroke_width
            if alignment == ALIGN_CENTER:
                x_offset += (textblk_w - line_widths[ii]) / 2
            txtblk_draw.text((x_offset, y_offset), line, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
            tmp_msk_draw.text((x_offset, y_offset), line, font=font, fill='white', stroke_width=stroke_width, stroke_fill='white')
            valid_bbox = tmp_msk.getbbox()
            if valid_bbox is None:
                continue
            txtblk_mask.paste(tmp_msk, mask=tmp_msk)
            tmp_msk.paste('black', [0, 0, tmp_msk.size[0],tmp_msk.size[1]])
            textpolygons.append([valid_bbox[0], valid_bbox[1], valid_bbox[2]-valid_bbox[0], valid_bbox[3]-valid_bbox[1]])
    bbox = txtblk_mask.getbbox()
    if bbox is None:
        return None, None, None
    textpolygons = np.array(textpolygons)
    textpolygons = xywh2xyxypoly(textpolygons)
    txtblk_img, txtblk_mask = txtblk_img.crop(bbox), txtblk_mask.crop(bbox)
    textpolygons[:, ::2] = np.clip(textpolygons[:, ::2] - bbox[0], 0, txtblk_mask.width-1)
    textpolygons[:, 1::2] = np.clip(textpolygons[:, 1::2] - bbox[1], 0, txtblk_mask.height-1)
    if rotation != 0:
        center = (txtblk_img.width/2, txtblk_img.height/2)
        txtblk_img = txtblk_img.rotate(rotation, Image.BICUBIC, expand=1)
        txtblk_mask = txtblk_mask.rotate(rotation, Image.BICUBIC, expand=1)
        new_center = (txtblk_img.width / 2, txtblk_img.height / 2)
        textpolygons = rotate_polygons(center, textpolygons, rotation, new_center)
    # txtblk_img, txtblk_mask = txtblk_img.crop(bbox), txtblk_mask.crop(bbox)
    # textpolygons[:, ::2] = np.clip(textpolygons[:, ::2] - bbox[0], 0, txtblk_mask.width-1)
    # textpolygons[:, 1::2] = np.clip(textpolygons[:, 1::2] - bbox[1], 0, txtblk_mask.height-1)
    return txtblk_img, txtblk_mask, textpolygons

def create_random_sampler(value, prob):
    if isinstance(prob, list):
        prob = np.array(prob).astype(np.float32)
    prob /= prob.sum()
    sampler = lambda : np.random.choice(value, replace=False, p=prob)
    return sampler

class ScaledSampler:
    def __init__(self, func_args, func='default'):
        if func == 'default':
            self.sampler_func = create_random_sampler(**func_args)
        else:
            raise NotImplementedError()
        pass
    def __call__(self, scaler=None, to_int=True):
        value = self.sampler_func()
        if scaler is not None:
            value = scaler * value
            if to_int:
                value = int(round(value))
        return value
        pass

class RandColorSampler:
    def __init__(self, func_args, func='default'):
        if func == 'default':
            self.sampler_func = create_random_sampler(**func_args)
        else:
            raise NotImplementedError()
        pass
    def __call__(self, scaler=None):
        value = self.sampler_func()
        if value == 'random':
            return (random.randint(0,255), random.randint(0,255), random.randint(0,255), 255)
        return value

class TextLinesSampler:
    def __init__(self, page_size, sampler_dict):
        self.page_w, self.page_h = page_size
        self.lang = sampler_dict['lang']
        self.lang_dict = load_dict(lang=self.lang)
        self.orientation_sampler = ScaledSampler(sampler_dict['orientation'])
        self.numlines_sampler = ScaledSampler(sampler_dict['num_lines'])
        self.length_sampler = ScaledSampler(sampler_dict['length'])
        self.min_num_lines = sampler_dict['min_num_lines']
        self.min_length = sampler_dict['min_length']
        self.alignment_sampler = create_random_sampler(**sampler_dict['alignment'])
        self.rotation_sampler = create_random_sampler(**sampler_dict['rotation'])
        
    def __call__(self, page_w=None, page_h=None, font_size=1):
        if page_w == None:
            page_w = self.page_w
        if page_h == None:
            page_h = self.page_h
        orientation = self.orientation_sampler()
        rotation = self.rotation_sampler()
        if rotation != 0:
            rotation = random.randint(-rotation, rotation)
        num_lines = max(self.numlines_sampler(page_h/font_size), self.min_num_lines)
        num_lines = random.randint(self.min_num_lines, num_lines)
        max_length = max(self.length_sampler(page_h/font_size), self.min_length)

        textlines = []
        dict_len = len(self.lang_dict)
        for ii in range(num_lines):
            line = ''
            length = random.randint(self.min_length, max_length)
            for jj in range(length):
                line += self.lang_dict[random.randrange(dict_len)] + ' '
            textlines.append(line[:length])
        return textlines, orientation, self.alignment_sampler(), rotation

class FontSampler:
    def __init__(self, font_dict, page_size) -> None:
        font_statics = font_dict['font_statics']
        font_dir = font_dict['font_dir']
        self.page_size = page_size

        self.size_sampler = ScaledSampler(font_dict['size'])
        self.color_sampler = RandColorSampler(font_dict['color'])
        self.sw_sampler = ScaledSampler(font_dict['stroke_width'])
        
        self.font_dir = font_dir
        self.sampler_range = font_dict['num']
        self.font_idx = 0

        font_statics = pd.read_csv(font_statics)
        self.font_list = list()
        for fontname in font_statics['font']:
            if osp.exists(osp.join(self.font_dir, fontname)):
                self.font_list.append(fontname)
                if len(self.font_list) >= self.sampler_range:
                    break
        assert len(self.font_list) > 0

    def __call__(self, page_size = None):
        if page_size is None:
            page_size = self.page_size
        page_w, page_h = page_size
        fontsize = self.size_sampler(page_h)
        stroke_width = self.sw_sampler(fontsize)
        color = self.color_sampler()
        if color == 'black':
            sw_color = (255, 255, 255, 255)
        elif color == 'white':
            sw_color = (0, 0, 0, 255)
        else:
            sw_color = self.color_sampler()
        # while (True):
        #     self.font_idx = random.randrange(0, self.sampler_range)
        #     fontname = self.font_statics.iloc[self.font_idx]['font']
        #     font_path = osp.join(self.font_dir, fontname)
        #     if osp.exists(font_path):
        #         break
        self.font_idx = random.randrange(0, self.sampler_range) % len(self.font_list)
        font_path = osp.join(self.font_dir, self.font_list[self.font_idx])
        font = ImageFont.truetype(font_path, fontsize)
        
        return font, color, stroke_width, sw_color


class TextBlkSampler:
    def __init__(self, page_size, max_tries, bboxlist=[]):
        self.page_w, self.page_h = page_size
        self.bboxlist = bboxlist
        self.max_tries = max_tries
        self.max_padding = int(round(0.05 * self.page_h))

    def __call__(self, bbox_w, bbox_h, padding=0, page_size=None):
        padding = int(round(padding))
        if page_size is not None:
            page_w, page_h = page_size
        else:
            page_w, page_h = self.page_w, self.page_h
        padding = min(self.max_padding, padding)
        bbox_w += 2*padding
        bbox_h += 2*padding
        x_range = page_w-bbox_w-1
        y_range = page_h-bbox_h-1
        if x_range < 0 or y_range < 0:
            return None
        for ii in range(self.max_tries):
            x, y = random.randint(0, x_range), random.randint(0, y_range)
            bbox_padded = [x, y, x + bbox_w, y + bbox_h]
            collide = False
            for bbox_exist in self.bboxlist:
                if union_area(bbox_exist, bbox_padded) > 0:
                    collide = True
                    break
            if not collide:
                break
        if not collide:
            bbox = [bbox_padded[0]+padding, bbox_padded[1]+padding, bbox_padded[2]-padding, bbox_padded[3]-padding]
            # bbox = [int(bb) for bb in bbox]
            self.bboxlist.append(bbox)
            return bbox
        return None

    def initialize(self, page_w, page_h, bboxlist=None, to_xywh=False):
        if bboxlist is None:
            self.bboxlist = []
        else:
            if to_xywh:
                self.bboxlist = yolo_xywh2xyxy(bboxlist, page_w, page_h)
                if self.bboxlist is not None:
                    self.bboxlist = self.bboxlist.tolist()
                else:
                    self.bboxlist = []
                

LANG_DICT = {'en': 0, 'ja': 1}
def lang2cls(lang: str) -> int:
    return LANG_DICT[lang]
def cls2lang(cls: int) -> str:
    return list(LANG_DICT.keys())[cls]

def get_max_var_color(mean_bgcolor):
    color_canditate = np.clip(np.array([mean_bgcolor-127, mean_bgcolor+127]), 0, 255).astype(np.int64)
    max_var_color = [c[0] if abs(c[0]-mean_bgcolor[ii]) > abs(c[1]-mean_bgcolor[ii]) else c[1] for ii, c in enumerate(zip(color_canditate[0], color_canditate[1]))]
    max_var_color = (max_var_color[0], max_var_color[1], max_var_color[2])
    return max_var_color


class ComicTextSampler:
    def __init__(self, page_size, sampler_dict, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.page_size = page_size
        self.num_txtblk = sampler_dict['num_txtblk']
        self.font_dict = sampler_dict['font']
        self.text_dict = sampler_dict['text']
        
        self.textlines_sampler = TextLinesSampler(page_size, sampler_dict['text'])
        self.font_sampler = FontSampler(self.font_dict, self.page_size)
        self.textblk_sampler = TextBlkSampler(page_size, max_tries=20)

        self.lang = sampler_dict['text']['lang']

    def drawtext_one_page(self, page_size=None, bboxlist=None, im_in=None, adaptive_color=False):
        if page_size is not None:
            page_w, page_h = page_size
        else:
            page_w, page_h = self.page_size
        if im_in is None:
            canvas = Image.new("RGBA", (page_w, page_h), 'white')
        else:
            canvas = Image.fromarray(cv2.cvtColor(im_in, cv2.COLOR_BGR2RGB))
            page_w, page_h = canvas.width, canvas.height
        canvas_msk = Image.new("L", (page_w, page_h), 'black')
        canvas_draw = ImageDraw.Draw(canvas)
        block_dicts = {}
        yolo_labels = []
        textpolylines = []
        self.textblk_sampler.initialize(page_w, page_h, bboxlist, True)
        for ii in range(self.num_txtblk):
            font, color, stroke_width, sw_color = self.font_sampler(page_size=self.page_size)
            textlines, orientation, alignment, rotation = self.textlines_sampler(font_size=font.size)
            txtblk_img, txtblk_mask, textpolygons = draw_textblk(textlines, font, fill=color, stroke_width=stroke_width, stroke_fill=sw_color, orientation=orientation, alignment=alignment, rotation=rotation)
            if txtblk_mask is None:
                continue
            bbox = self.textblk_sampler(txtblk_img.width, txtblk_img.height, font.size*1.2, page_size=(page_w, page_h))
            if bbox is not None:
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + txtblk_mask.width, bbox[1] + txtblk_mask.height
                re_draw = False
                if im_in is not None:
                    mean_bgcolor = np.mean(im_in[y1: y2, x1: x2], axis=(0, 1))
                    max_var_color = get_max_var_color(mean_bgcolor)
                    # color_canditate = np.clip(np.array([mean_bgcolor-127, mean_bgcolor+127]), 0, 255).astype(np.int64)
                    # max_var_color = [c[0] if abs(c[0]-mean_bgcolor[ii]) > abs(c[1]-mean_bgcolor[ii]) else c[1] for ii, c in enumerate(zip(color_canditate[0], color_canditate[1]))]
                    # max_var_color = (max_var_color[0], max_var_color[1], max_var_color[2])
                    if color == 'black':
                        color_rep = np.array([0, 0, 0])
                    elif color == 'white':
                        color_rep = np.array([255, 255, 255])
                    else:
                        color_rep = np.array(color[:3])
                    color_var = np.sum(np.abs(mean_bgcolor - color_rep))
                    if not adaptive_color:
                        if color_var < 127:
                            color = max_var_color

                            sw_color = get_max_var_color(np.array(color))
                            re_draw = True
                    else:
                        color = max_var_color
                        sw_color = get_max_var_color(np.array(color))
                        re_draw = True
                if stroke_width != 0 and im_in is not None:
                    # sw_color = get_max_var_color(color)
                    re_draw = True
                if re_draw:
                    txtblk_img, txtblk_mask, textpolygons = draw_textblk(textlines, font, fill=color, stroke_width=stroke_width, stroke_fill=sw_color, orientation=orientation, alignment=alignment, rotation=rotation)
                blk_dict = {
                    'lang': self.lang, 
                    'lang_cls': lang2cls(self.lang),
                    'xyxy': [x1, y1, x2, y2],
                    'polylines': textpolygons
                }
                block_dicts[str(ii)+'-'+self.lang] = blk_dict
                textpolygons[:, ::2] += x1
                textpolygons[:, 1::2] += y1
                textpolylines += textpolygons.astype(np.int64).tolist()
                yolo_labels += [[x1, y1, x2, y2]]
                canvas.paste(txtblk_img, (bbox[0], bbox[1]), mask=txtblk_mask)
                canvas_msk.paste(txtblk_mask, (bbox[0], bbox[1]), mask=txtblk_mask)

        rst = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
        rst_msk = np.array(canvas_msk)
        yolo_labels = xyxy2yolo(np.array(yolo_labels), page_w, page_h)
        if yolo_labels is not None:
            cls = np.ones((yolo_labels.shape[0], 1)) * lang2cls(self.lang)
            yolo_labels = np.concatenate((cls, yolo_labels), axis=1)
        return rst, rst_msk, block_dicts, yolo_labels, np.array(textpolylines)

def render_comictext(comic_sampler_list, img_dir, label_dir=None, render_num=700, save_dir=None, save_prefix=None, show=False):
    if osp.exists(osp.join(img_dir, 'statistics.csv')):
        statistics = pd.read_csv(osp.join(img_dir, 'statistics.csv'))
    else:
        statistics = None
        imglist = find_all_imgs(img_dir)
        # render_num = min(render_num, len(imglist))
    num_im = len(imglist)
    for ii in tqdm(range(render_num)):
        im_idx = ii % num_im
        if statistics is not None:
            imgname = statistics.loc[im_idx]['name']
        else:
            imgname = imglist[im_idx]
        img = imread(osp.join(img_dir, imgname))
        cs_idx = ii % len(comic_sampler_list)
        bboxlist = []
        labels = None
        if label_dir is not None:
            labelname = imgname.replace(pathlib.Path(imgname).suffix, '.txt')
            label_path = osp.join(label_dir, labelname)
            labels = np.loadtxt(label_path)
            if len(labels) != 0:
                if len(labels.shape) == 1:
                    labels = np.array([labels])
                clslist, bboxlist = labels[:, 0], np.copy(labels[:, 1:])
            else:
                labels = None
                bboxlist = []
        rst, rst_msk, block_dicts, yolo_labels, textpolylines = comic_sampler_list[cs_idx].drawtext_one_page(im_in=img, bboxlist=bboxlist, adaptive_color=True)
        if save_dir is not None:
            if save_prefix is not None:
                save_name = save_prefix + '{0:09d}'.format(ii) + '.jpg'
            else:
                save_name = 'syn-' + imgname
            yolo_save_path = osp.join(save_dir, save_name.replace(pathlib.Path(save_name).suffix, '.txt'))
            content = ''
            if yolo_labels is not None:
                if labels is None:
                    content = get_yololabel_strings(yolo_labels[:, 0], yolo_labels[:, 1:])
                else:
                    yolo_labels = np.concatenate((labels, yolo_labels))
                    content = get_yololabel_strings(yolo_labels[:, 0], yolo_labels[:, 1:])
            if content == '' and label_dir is not None:
                shutil.copy(label_path, yolo_save_path)
            else:
                with open(yolo_save_path, 'w', encoding='utf8') as f:
                    f.write(content)
                
            linepoly_save_path = osp.join(save_dir, 'line-'+osp.basename(yolo_save_path))
            np.savetxt(linepoly_save_path, textpolylines, fmt='%d')
            imwrite(osp.join(save_dir, save_name), rst, ext='.jpg')
            imwrite(osp.join(save_dir, 'mask-'+save_name), rst_msk)

        if show:
            for pts in textpolylines:
                rst = cv2.polylines(rst, [np.array(pts).reshape((-1, 1, 2))], color=(255, 0, 0), isClosed=True, thickness=2)
            cv2.imshow('rst', rst)
            cv2.waitKey(0)


if __name__ == '__main__':

    eng_sampler_dict = {
                    'num_txtblk': 20,
                    'font': {
                            'font_dir': 'data/fonts',
                            'font_statics': 'data/font_statics_en.csv',
                            'num': 500,
                            'size': {'value': [0.02, 0.03, 0.15],
                                    'prob': [1, 0.4, 0.15]},
                            'stroke_width': {'value': [0, 0.1, 0.15],
                                            'prob': [1, 0.2, 0.2]},
                            'color': {'value': ['black', 'random'],
                                        'prob': [1, 0.4]},
                    },
                    'text': {
                        'lang': 'en',
                        'orientation': {'value': [1, 0],
                                            'prob': [0, 1]},
                        'rotation': {'value': [0, 30, 60],
                                            'prob': [1, 0.3, 0.1]},
                        'num_lines': {'value': [0.15],
                                'prob': [1]}, 
                        'length': {'value': [1],
                                'prob': [1]},
                        'min_num_lines': 1,
                        'min_length': 3,
                        'alignment': {'value': [ALIGN_LEFT, ALIGN_CENTER],
                                'prob': [0.3, 1]}
                    }
                }

    ja_sampler_dict = {
                    'num_txtblk': 20,
                    'font': {
                            'font_dir': 'data/fonts',   # font file directory
                            'font_statics': 'data/font_statics_jp.csv',     # Just a font list to use, please create your own list and ignore the last two cols.
                            'num': 500,     # first 500 of the fontlist will be used 
                            # params to 
                            'size': {'value': [0.02, 0.03, 0.15],
                                    'prob': [1, 0.4, 0.15]},
                            'stroke_width': {'value': [0, 0.1, 0.15],
                                            'prob': [1, 0.5, 0.2]},
                            'color': {'value': ['black', 'white', 'random'],
                                        'prob': [1, 1, 0.4]},
                    },
                    'text': {
                        'lang': 'ja',   # render japanese, 'en' for english
                        'orientation': {'value': [1, 0],    # 1 is vertical text.
                                            'prob': [1, 0.3]},
                        'rotation': {'value': [0, 30, 60],
                                            'prob': [1, 0.3, 0.1]},
                        'num_lines': {'value': [0.15],
                                'prob': [1]}, 
                        'length': {'value': [0.3],
                                'prob': [1]},
                        'min_num_lines': 1,
                        'min_length': 3,
                        'alignment': {'value': [ALIGN_LEFT, ALIGN_CENTER],
                                'prob': [0.3, 1]}
                    }
                }



    # random.seed(0)
    # cts = ComicTextSampler((845, 1280), sampler_dict, seed=0)
    # jp_cts = ComicTextSampler((845, 1280), ja_sampler_dict, seed=0)
    
    # img_dir = r'../../datasets/pixanimegirls'
    # save_dir = r'../../datasets/pixanimegirls/processed'
    # os.makedirs(save_dir, exist_ok=True)

    # img_dir = r'../../datasets/ComicErased'
    # label_dir = img_dir
    # save_dir = r'../../datasets/ComicErased/processed'
    # os.makedirs(save_dir, exist_ok=True)
    # render_comictext([jp_cts, cts], img_dir, save_dir=save_dir, save_prefix=None, render_num=4000, label_dir=None)


