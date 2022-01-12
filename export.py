
from cv2 import imshow
import numpy as np
import onnxruntime
import cv2
import torch
import onnx
from basemodel import TextDetBase
import onnxsim
from models.common import Conv
from models.yolo import Detect
import torch.nn as nn
import time
from dataset import letterbox

class SiLU(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def export_onnx(model, im, file, opset, train=False, simplify=True, dynamic=False, inplace=False):
    # YOLOv5 ONNX export
    f = file + '.onnx'
    for k, m in model.named_modules():
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = inplace
            m.onnx_dynamic = False
    torch.onnx.export(model, im, f, verbose=False, opset_version=opset,
                        training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                        do_constant_folding=not train,
                        input_names=['images'],
                        output_names=['output', 'dummy1', 'dummy2', 'dummy3', 'seg', 'det'],
                        dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                    'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
                                    } if dynamic else None)

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    import onnxsim

    model_onnx, check = onnxsim.simplify(
        model_onnx,
        dynamic_input_shape=dynamic,
        input_shapes={'images': list(im.shape)} if dynamic else None)
    assert check, 'assert check failed'
    onnx.save(model_onnx, f)

if __name__ == '__main__':
    batch_size, imgsz = 1, 1024
    device = 'cpu'
    im = torch.zeros(batch_size, 3, imgsz, imgsz).to(device)
    model_path = r'data/textdetector.pt'
    model = TextDetBase(model_path).to(device)

    # export_onnx(model, im, model_path, 11)

    img_path = r'dataset\train\manga-2217.jpg'
    img = cv2.imread(img_path)
    img, ratio, (dw, dh) = letterbox(img, new_shape=(1024, 1024), auto=False, stride=64)
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray([img]) / 255.
    
    model_path = r'data\textdetector.pt.onnx'
    # net = cv2.dnn.readNetFromONNX(model_path)

    cuda = False
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(model_path, providers=providers)

    t0 = time.time()
    outnames = session.get_outputs()

    # print(session.get_outputs()[[0, 2]])
    y = session.run([outnames[0].name, outnames[4].name, outnames[5].name], {session.get_inputs()[0].name: img.astype(np.float32)})
    
    print(f'{time.time() - t0}')
    print(y)