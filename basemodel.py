from utils.general import CUDA, DEVICE
from models.yolo import Model
import torch
import cv2
import numpy as np
from models.yolo import load_yolov5
import glob
import torch.nn as nn
from utils.weight_init import init_weights
from models.common import C3, Conv
from torchsummary import summary
import torch.nn.functional as F
import copy

TEXTDET_MASK = 0
TEXTDET_DET = 1
TEXTDET_INFERENCE = 2

class double_conv_up_c3(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, act=True):
        super(double_conv_up_c3, self).__init__()
        self.conv = nn.Sequential(
        C3(in_ch+mid_ch, mid_ch, act=act),
        nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=4, stride = 2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class double_conv_c3(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, act=True):
        super(double_conv_c3, self).__init__()
        if stride > 1 :
            self.down = nn.AvgPool2d(2,stride=2) if stride > 1 else None
        self.conv = C3(in_ch, out_ch, act=act)

    def forward(self, x):
        if self.down is not None :
            x = self.down(x)
        x = self.conv(x)
        return x

# class UnetHead(nn.Module):
#     def __init__(self, act=True) -> None:

#         super(UnetHead, self).__init__()
#         self.down_conv1 = double_conv_c3(512, 512, 2, act=act)
#         self.down_conv2 = double_conv_c3(512, 512, 2, act=act)
#         self.upconv0 = double_conv_up_c3(0, 512, 256, act=act)
#         self.upconv1 = double_conv_up_c3(256, 512, 256, act=act)
#         self.upconv2 = double_conv_up_c3(256, 512, 256, act=act)
#         self.upconv3 = double_conv_up_c3(0, 512, 256, act=act)
#         self.upconv4 = double_conv_up_c3(128, 256, 128, act=act)
#         self.upconv5 = double_conv_up_c3(64, 128, 64, act=act)
#         # self.conv_mask = C3(64, 32, act=act)
#         self.upconv6 = nn.Sequential(
#             nn.PixelShuffle(2),
#             nn.Conv2d(8, 1, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def forward(self, f160, f80, f40, f20, f3, forward_mode=TEXTDET_MASK):
#         # input: 640@3
#         d10 = self.down_conv1(f3) # 512@10
#         d5 = self.down_conv2(d10) # 512@5
#         u10 = self.upconv0(d5)  # 256@10
#         u20 = self.upconv1(torch.cat([u10, d10], dim = 1)) # 256@20
#         u40 = self.upconv2(torch.cat([f20, u20], dim = 1)) # 256@40

#         if forward_mode == TEXTDET_DET:
#             return f80, f40, u40
#         else:
#             u80 = self.upconv3(torch.cat([f40, u40], dim = 1)) # 256@80
#             u160 = self.upconv4(torch.cat([f80, u80], dim = 1)) # 128@160
#             u320 = self.upconv5(torch.cat([f160, u160], dim = 1)) # 64@320
#             u320 = self.conv_mask(u320)
#             mask = self.upconv6(u320)
#             if forward_mode == TEXTDET_MASK:
#                 return mask
#             else:
#                 return mask, [f80, f40, u40]
            
#     def init_weight(self, init_func):
#         self.apply(init_func)

class UnetHead(nn.Module):
    def __init__(self, act=True) -> None:

        super(UnetHead, self).__init__()
        self.down_conv1 = double_conv_c3(512, 512, 2, act=act)
        self.upconv0 = double_conv_up_c3(0, 512, 256, act=act)
        self.upconv2 = double_conv_up_c3(256, 512, 256, act=act)
        self.upconv3 = double_conv_up_c3(0, 512, 256, act=act)
        self.upconv4 = double_conv_up_c3(128, 256, 128, act=act)
        self.upconv5 = double_conv_up_c3(64, 128, 64, act=act)
        self.upconv6 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, f160, f80, f40, f20, f3, forward_mode=TEXTDET_MASK):
        # input: 640@3
        d10 = self.down_conv1(f3) # 512@10
        u20 = self.upconv0(d10)  # 256@10
        u40 = self.upconv2(torch.cat([f20, u20], dim = 1)) # 256@40

        if forward_mode == TEXTDET_DET:
            return f80, f40, u40
        else:
            u80 = self.upconv3(torch.cat([f40, u40], dim = 1)) # 256@80
            u160 = self.upconv4(torch.cat([f80, u80], dim = 1)) # 128@160
            u320 = self.upconv5(torch.cat([f160, u160], dim = 1)) # 64@320
            mask = self.upconv6(u320)
            if forward_mode == TEXTDET_MASK:
                return mask
            else:
                return mask, [f80, f40, u40]
            
    def init_weight(self, init_func):
        self.apply(init_func)

class DBHead(nn.Module):
    def __init__(self, in_channels, k = 50, shrink_with_sigmoid=True):
        super().__init__()
        self.k = k
        self.shrink_with_sigmoid = shrink_with_sigmoid
        self.upconv3 = double_conv_up_c3(0, 512, 256)
        self.upconv4 = double_conv_up_c3(128, 256, 128)
        self.conv = nn.Sequential(
            nn.Conv2d(128, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2)
            )
        self.thresh = self._init_thresh(in_channels)

    def forward(self, f80, f40, u40, shrink_with_sigmoid=True, step_eval=False):
        shrink_with_sigmoid = self.shrink_with_sigmoid
        u80 = self.upconv3(torch.cat([f40, u40], dim = 1)) # 256@80
        x = self.upconv4(torch.cat([f80, u80], dim = 1)) # 128@160
        x = self.conv(x)
        threshold_maps = self.thresh(x)
        x = self.binarize(x)
        shrink_maps = torch.sigmoid(x)
        
        if self.training:
            binary_maps = self.step_function(shrink_maps, threshold_maps)
            if shrink_with_sigmoid:
                return torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
            else:
                return torch.cat((shrink_maps, threshold_maps, binary_maps, x), dim=1)
        else:
            if step_eval:
                return self.step_function(shrink_maps, threshold_maps)
            else:
                return torch.cat((shrink_maps, threshold_maps), dim=1)

    def init_weight(self, init_func):
        self.apply(init_func)

    def _init_thresh(self, inner_channels, serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels // 4, smooth=smooth, bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self, in_channels, out_channels, smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=True))
            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

class TextDetector(nn.Module):
    def __init__(self, weights, map_location='cpu', forward_mode=TEXTDET_MASK, act=True):
        super(TextDetector, self).__init__()
        yolov5s_backbone = load_yolov5(weights=weights, map_location=map_location)
        yolov5s_backbone.eval()
        out_indices = [1, 3, 5, 7, 9]
        yolov5s_backbone.out_indices = out_indices
        yolov5s_backbone.model = yolov5s_backbone.model[:max(out_indices)+1]
        self.seg_net = UnetHead(act=act)
        self.backbone = yolov5s_backbone
        self.dbnet = None
        self.forward_mode = forward_mode

    def train_mask(self):
        self.forward_mode = TEXTDET_MASK
        self.backbone.eval()
        self.seg_net.train()

    def initialize_db(self, unet_weights):
        self.dbnet = DBHead(64)
        self.seg_net.load_state_dict(torch.load(unet_weights, map_location='cpu')['weights'])
        self.dbnet.init_weight(init_weights)
        self.dbnet.upconv3 = copy.deepcopy(self.seg_net.upconv3)
        self.dbnet.upconv4 = copy.deepcopy(self.seg_net.upconv4)
        del self.seg_net.upconv3
        del self.seg_net.upconv4
        del self.seg_net.upconv5
        del self.seg_net.upconv6
        # del self.seg_net.conv_mask
    
    def train_db(self):
        self.forward_mode = TEXTDET_DET
        self.backbone.eval()
        self.seg_net.eval()
        self.dbnet.train()

    def forward(self, x):
        forward_mode = self.forward_mode
        with torch.no_grad():
            outs = self.backbone(x)
        if forward_mode == TEXTDET_MASK:
            return self.seg_net(*outs, forward_mode=forward_mode)
        elif forward_mode == TEXTDET_DET:
            with torch.no_grad():
                outs = self.seg_net(*outs, forward_mode=forward_mode)
            return self.dbnet(*outs)

def get_base_det_models(model_path, device='cpu', half=False):
    textdetector_dict = torch.load(model_path, map_location=device)
    blk_det = load_yolov5(textdetector_dict['blk_det'])
    text_seg = UnetHead()
    text_seg.load_state_dict(textdetector_dict['text_seg'])
    text_det = DBHead(64)
    text_det.load_state_dict(textdetector_dict['text_det'])
    if half:
        return blk_det.eval().half(), text_seg.eval().half(), text_det.eval().half()
    return blk_det.eval(), text_seg.eval(), text_det.eval()

class TextDetBase(nn.Module):
    def __init__(self, model_path, device='cpu', half=False):
        super(TextDetBase, self).__init__()
        self.blk_det, self.text_seg, self.text_det = get_base_det_models(model_path, device, half)   
    def forward(self, features):
        blks, features = self.blk_det(features, detect=True)
        mask, features = self.text_seg(*features, forward_mode=TEXTDET_INFERENCE)
        lines = self.text_det(*features, step_eval=True)
        return blks, mask, lines
        # return blks

if __name__ == '__main__':
    device = 'cuda'
    weights = r'data/yolov5sblk.pt'

    model = TextDetector(weights, map_location=DEVICE)
    model.to(DEVICE)
    model.train_mask()
    summary(model, (3, 640, 640), device=DEVICE)

    # model.initialize_db(unet_weights='data/unet_head.pt')
    # model.train_db()
    # summary(model, (3, 640, 640), device=DEVICE)


