import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.yolov5.yolo import load_yolov5_ckpt
from models.yolov5.common import C3, Conv
from torchsummary import summary
bn_mom = 0.1
from utils.weight_init import init_weights


class FeatureExtractorYoloV5s(nn.Module):
    '''YoloV5s feature extractor wrapper
    Args:
        out_indices: layer indices
                    - 1: 4@64       # given input 1,3,1024,1024, out dim is 1,64,1024/4,1024/4
                    - 2: 4@64
                    - 3: 8@128
                    - 4: 8@128
                    - 5: 16@256
                    - 6: 16@256
                    - 7: 32@512
                    - 8: 32@512
                    - 9: 32@512

    Returns:
        a list of feature maps      
    '''

    def __init__(self, weight_path: str, out_layer_indices: List[int], remove_notused=True) -> None:
        super(FeatureExtractorYoloV5s, self).__init__()
        weights = load_yolov5_ckpt(weight_path, map_location='cpu', out_indices=out_layer_indices)
        if remove_notused:
            weights.model = weights.model[:max(out_layer_indices)+1]
        self.weights = weights

    def forward(self, x):
        return self.weights(x)

class downconvc3(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, act=True):
        super(downconvc3, self).__init__()
        if stride > 1 :
            self.down = nn.AvgPool2d(2,stride=2) if stride > 1 else None
        self.conv = C3(in_ch, out_ch, act=act, final_act=False)

        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.down is not None :
            x = self.down(x)
        return self.conv(x)


class DualPath(nn.Module):
    def __init__(self, hc: int, lc: int, act=True, downsample=None) -> None:
        super(DualPath, self).__init__()
        div = lc // hc
        # https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two
        assert (div & (div-1) == 0) and div != 0    # div must be power of 2
        h2l = [C3(hc, hc, act=act)]
        hc_tmp = hc
        for ii in range(int(math.log(div, 2)))[:-1]:
            h2l.append(Conv(hc_tmp, hc_tmp * (2 ** (ii + 1)), k=3, s=2, act=act))
            hc_tmp = hc_tmp * (2 ** (ii + 1))
        h2l.append(Conv(hc_tmp, lc, k=3, s=2, act=False))
        self.h2l = nn.Sequential(*h2l)
        self.downsample = nn.Identity() if downsample is None else downsample

        self.l2h = nn.Sequential(
            C3(lc, lc, act=act),
            Conv(lc, hc, k=3, s=1, act=False),
        )
    
    def forward(self, hmap: torch.tensor, lmap: torch.tensor):
        hdim = hmap.shape[-1]
        return hmap + F.interpolate(self.l2h(lmap), size=(hdim, hdim), mode='bilinear', align_corners=True), \
            self.downsample(lmap + self.h2l(hmap))

class BnConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ks: int = 1):
        super(BnConv, self).__init__()
        self.bn = nn.BatchNorm2d(in_ch, momentum=bn_mom)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_ch, out_ch, ks, bias=False)
    def forward(self, x):
        return self.conv(self.act(self.bn(x)))

def avgpool_bnconv(in_ch: int, out_ch: int, k: int, s: int):
    return nn.Sequential(
        nn.AvgPool2d(k, s, s),
        BnConv(in_ch, out_ch, ks=1)
    )

class ScaleFuseBranch(nn.Module):
    def __init__(self, in_ch: int, branch_ch: int, k: int, s: int):
        self.avgbnconv = avgpool_bnconv(in_ch, branch_ch, k, s)
        self.fuseconv = BnConv(branch_ch, branch_ch, 1)

    def forward(self, residual, x):
        _, _, h, w = residual.shape
        x = F.interpolate(self.avgbnconv(x), [h, w], mode='bilinear')
        return self.fuseconv(residual + x)
    
class DAPPMEX(nn.Module):
    def __init__(self, in_ch, branch_ch, out_ch, num_scale: int = 5):
        super(DAPPMEX, self).__init__()
        self.scale_list = nn.ModuleList()
        self.scale_list.add_module(f'scale-1', BnConv(in_ch, out_ch, 1))
        for ii in range(num_scale - 1):
            i = ii + 2
            self.scale_list.add_module(
                f'scale-{i}',
                ScaleFuseBranch(in_ch, branch_ch, 2**i + 1, 2**(i - 1))
            )
        self.scale_list.add_module(f'scale-{num_scale}', ScaleFuseBranch(in_ch, branch_ch, 1, 1))
        self.shortcut = BnConv(in_ch, out_ch)
        self.compression = BnConv(branch_ch * num_scale, out_ch)

    def forward(self, x):
        x_list = []
        x_list.append(self.scale_list[0](x))
        for layer in self.scale_list[1:]:
            x_list.append(layer(x, x_list[-1]))
        return self.compression(torch.cat(x_list, 1)) + self.shortcut(x)

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    nn.BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    nn.BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    nn.BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    nn.BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    nn.BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    nn.BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process3 = nn.Sequential(
                                    nn.BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    nn.BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )        
        self.compression = nn.Sequential(
                                    nn.BatchNorm2d(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    nn.BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

    def forward(self, x):

        #x = self.downsample(x)
        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear')+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear')+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear')+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear')+x_list[3])))
       
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out 
            

class DDRSegNet(nn.Module):
    # Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes
    # https://arxiv.org/abs/2101.06085
    def __init__(self,
                 hmap_scale: int,
                 final_scale: int,
                 hmap: torch.tensor, 
                 lmap: torch.tensor, 
                 head_ch: int,
                 num_classes: int = 1,
                 spp_ch: int = 128,
                 act=True) -> None:
        super(DDRSegNet, self).__init__()
        self.hmap_scale = hmap_scale
        _, hc, hres, _ = hmap.shape
        _, lc, lres, _ = lmap.shape
        full_res = hres * hmap_scale
        final_res = full_res // final_scale
        self.path_layers = nn.ModuleList()
        while lres > final_res:
            lres //= 2
            dp = DualPath(hc, lc, act=act, downsample=downconvc3(lc, lc * 2, stride=2, act=act))
            hmap, lmap = dp(hmap, lmap)
            hc, lc = hmap.shape[1], lmap.shape[1]
            self.path_layers.append(dp)
        self.aux_head = self._make_seghead(hc, head_ch, num_classes, act=act)
        self.hpath_expand = Conv(hc, hc * 2, k=1, s=1, act=False)
        hc *= 2

        assert lres == final_res
        self.head = self._make_seghead(hc, head_ch, num_classes, act=act)
        self.dappm = DAPPM(lc, spp_ch, hc)
        self.apply(init_weights)


    def _make_seghead(self, in_ch, mid_ch, out_ch, act=True):
        return nn.Sequential(
            Conv(in_ch, mid_ch, k=3, s=1, act=act),
            Conv(mid_ch, out_ch, k=1, s=1, act=act)
        )

    def forward(self, hmap: torch.tensor, lmap: torch.tensor):
        _, hc, hres, _ = hmap.shape
        full_res = hres * self.hmap_scale
        for ii, layer in enumerate(self.path_layers):
            hmap, lmap = layer(hmap, lmap)

        if self.training:
            aux_map = F.interpolate(self.aux_head(hmap), size=[full_res, full_res], mode='bilinear')
        hmap = self.hpath_expand(hmap)
        lmap = F.interpolate(self.dappm(lmap), size=[hres, hres], mode='bilinear')
        hmap = F.interpolate(self.head(hmap + lmap), size=[full_res, full_res], mode='bilinear')
        if self.training:
            return hmap, aux_map
        else:
            return hmap
    

class TextDetectorV2(nn.Module):
    def __init__(self, 
                 feature_extractor: nn.Module, 
                 seg_net: nn.Module) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.seg_net = seg_net
    def forward(self, x):

        x = self.feature_extractor(x)
        x = self.seg_net(*x)
        return x

def build_text_detector(device='cpu'):

    dummy_tensor = torch.randn((2, 3, 1024, 1024)).to(device)

    # maybe we should use a different obj detector such as yolox or remove it, idk
    v5weights = r'data/yolov5sblk.ckpt'
    fe = FeatureExtractorYoloV5s(v5weights, out_layer_indices=[3, 5])
    features = fe(dummy_tensor)
    hmap, lmap = features[0], features[1]
    hscale = dummy_tensor.shape[-1] // hmap.shape[-1]
    segnet = DDRSegNet(hscale, 64, hmap, lmap, head_ch=256, num_classes=1, act=True)
    model = TextDetectorV2(fe, segnet)
    return model
    # segnet.to(device)
    # segnet(hmap, lmap)


if __name__ == '__main__':
    DEVICE = 'cpu'
    model = build_text_detector()
    model.to(DEVICE)
    summary(model, (3, 1024, 1024), device=DEVICE)

    # with torch.no_grad():
    #     model.eval()
    #     input = torch.randn((1, 3, 1024, 1024))
    #     import time
    #     t0 = time.time()
    #     out = model(input)
    #     print(time.time() - t0)
    # pass

    d = DAPPMEX(256, 128, 128)
    print(d)