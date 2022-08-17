from models import AE, AEloss
from models.AE import AE
from models.vgg import DFF, SiameseVgg19Net

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torch
from torchvision.transforms import Normalize

import torchvision.transforms as Transforms
from PIL import Image


class Inference(nn.Module):
    def __init__(self):
        super(Inference, self).__init__()
        self.ae = AE()

    def re_size(self, Fd, Fae):
        out = Fd - Fae
        out = out.detach().cpu().numpy()
        return out


transform = Transforms.Compose([
    # Transforms.Resize((720, 720)),
    Transforms.ToTensor()
    # Transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

img_transform = Transforms.ToTensor()

if __name__ == '__main__':
    # mat1 = cv2.imread('./data/mvtec/grid/test/metal_contamination/000.png', 1)
    mat1 = cv2.imread('./models/000.png')
    src = cv2.resize(mat1, (720, 720))
    mat1 = transform(mat1)

    mat1 = torch.unsqueeze(mat1, dim=0)
    mat2 = mat1

    Inference = Inference()
    SiameseVgg19Net = SiameseVgg19Net()
    SiameseVgg19Net.eval()
    dff = DFF()
    ae = torch.load('./weight/weight/ae50.pt', map_location='cpu')
    ae.eval()
    # grad =
    print(ae.state_dict().keys())
    value = SiameseVgg19Net(mat1, mat2)
    value = dff.Cat(value)
    out = ae(value[0], value[1])

    inf = Inference.re_size(value[0], out[0])
    inf = np.mean(inf, axis=1)
    inf /= np.max(inf)
    inf = 1.0 - inf
    inf = np.squeeze(inf, axis=0)

    inf = cv2.resize(inf, (720, 720), interpolation=cv2.INTER_LINEAR)
    inf = np.uint8(255 * inf)

    inf = cv2.applyColorMap(inf, cv2.COLORMAP_JET)
    src = src.astype('uint8')
    out = np.hstack((src, inf))
    cv2.imshow('11', out)
    cv2.waitKey(0)
    print(1)
