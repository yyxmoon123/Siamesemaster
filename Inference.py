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



def read_mat(path):
    mat1 = cv2.imread(path, 2).astype('float32')
    mat1 = cv2.cvtColor(mat1, cv2.COLOR_GRAY2BGR)
    mat1 = mat1[None]
    tor1 = torch.from_numpy(mat1)
    tor1 = tor1.permute(0, 3, 1, 2)
    return tor1


def infer(FAEs, FDs):
    pass


transform = Transforms.Compose([
    Transforms.Resize((720, 720)),
    Transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    Transforms.ToTensor(),
])


if __name__ == '__main__':
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # SimaseNet = SiameseVgg19Net()
    # SimaseNet = SimaseNet.to(device)
    # mat1Path = './data/3.jpg'
    # mat2Path = './data/4.jpg'
    # mat1  = torch.from_numpy(cv2.imread(mat1Path,1))
    # mat1 = transform(mat1).astype('float32').to(device)
    # mat2 = cv2.imread(mat2Path,1)
    # mat2 = transform(mat2).to(device)
    # values = SimaseNet(mat1, mat2)
    ae = AE()
    SiameseVgg19Net = SiameseVgg19Net()
    
