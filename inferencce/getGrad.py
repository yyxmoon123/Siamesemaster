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


def readimg(path):
    mat1 = cv2.imread(path, flags=1)
    infer_transform = Transforms.Compose([
        Transforms.ToTensor(),
        Transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    img1 = infer_transform(mat1)
    img1 = torch.unsqueeze(img1, dim=0)
    img1 = torch.autograd.Variable(img1,requires_grad=True)
    img2 = img1
    return img1, img2


def hook_backward_fn(module, grad_input, grad_output):
    print(f"module: {module}")
    print(f"grad_output: {grad_output}")
    print(f"grad_input: {grad_input}")
    print("*" * 20)


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(5504, 2952, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(2952)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(2952, 400, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(400)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(400, 200, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(200),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(200, 400, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(400),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(400, 2952, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(2952),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(2952, 5504, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(5504),
        )
        self._init_weight()
        self.conv6.register_backward_hook(hook_backward_fn)
        self.gradients = None

    def branch(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        add1 = x
        x = F.relu(self.conv2(x), inplace=True)
        add0 = x
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x) + add0, inplace=True)
        x = F.relu(self.conv5(x) + add1, inplace=True)
        x = F.relu(self.conv6(x), inplace=True)
        return x

    def forward(self, x1, x2):
        Faes = self.branch(x1)
        Faeg = self.branch(x2)
        return Faes, Faeg

    def load_exsit(self,f):
        net = torch.load(f=f,map_location='cpu')
        for keys in net.state_dict().items():
            print()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def gradients_hook(self,grad):
        self.gradients = grad

    def get_gradients(self):
        return self.gradients

if __name__ == '__main__':
    img1, img2 = readimg('../models/008.png')
    SiameseVgg19Net = SiameseVgg19Net()
    SiameseVgg19Net.eval()
    net = torch.load('../weight/weight/ad41.pt', map_location='cpu')
    net.eval()
    ae = AE()
    ae.eval()
    dff = DFF()
    ae.load_exsit('../weight/weight/ad41.pt')

