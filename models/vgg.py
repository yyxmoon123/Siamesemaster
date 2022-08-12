import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate

from collections import OrderedDict

import torchvision.models as model
import torchvision.models as models
import math
import numpy as np
import h5py as h5
from torchvision.transforms import Resize
import torch.nn.init as init
from torch.autograd import Variable


class SiameseVgg19Net(nn.Module):
    def __init__(self):
        super(SiameseVgg19Net, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            # 1层
            ('conv1_1', nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('max1', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
            # 2layer
            ('conv2_1', nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu2_1', nn.ReLU(inplace=True)),
            ('conv2_2', nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu2_2', nn.ReLU(inplace=True)),
            ('max2', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
            # 3 layer
            ('conv3_1', nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu3_1', nn.ReLU(inplace=True)),
            ('conv3_2', nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu3_2', nn.ReLU(inplace=True)),
            ('conv3_3', nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu3_3', nn.ReLU(inplace=True)),
            ('conv3_4', nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu3_4', nn.ReLU(inplace=True)),
            ('max3', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
            # 4layer
            ('conv4_1', nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu4_1', nn.ReLU(inplace=True)),
            ('conv4_2', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu4_2', nn.ReLU(inplace=True)),
            ('conv4_3', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu4_3', nn.ReLU(inplace=True)),
            ('conv4_4', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu4_4', nn.ReLU(inplace=True)),
            ('max4', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
            # 5 layer
            ('conv5_1', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu5_1', nn.ReLU(inplace=True)),
            ('conv5_2', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu5_2', nn.ReLU(inplace=True)),
            ('conv5_3', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu5_3', nn.ReLU(inplace=True)),
            ('conv5_4', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu5_4', nn.ReLU(inplace=True))
        ])
        )

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        #### sb论文 写都写不清楚，图也画不好
        mod = models.vgg19(pretrained=True)
        for i in range(len(self.model.state_dict().items())):
            # print(list(mod.state_dict().items())[i][0], '\t', list(self.model.state_dict().items())[i][0])
            list(self.model.state_dict().items())[i][1][:] = list(mod.state_dict().items())[i][1][:]

    def branch(self, input):
        out = []
        for index, MO in enumerate(self.model):
            input = MO(input)
            if index in {1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26, 29, 31, 33, 35}:
                out.append(input)
        return out

    def forward(self, Ig, Is):
        out1 = self.branch(Ig)
        out2 = self.branch(Is)

        return {'Fs': out2, 'Fd': out1}


class DFF():
    def __init__(self):
        pass

    def ReSIZE(self, tens):
        value = tens
        # print('resize前', value.shape)
        s1, s2, s3, s4 = value.shape[0], value.shape[1], value.shape[2], value.shape[3]
        if s3 >= 128:
            value = interpolate(value, size=(128, 128))
        elif s3 < 128:
            value = nn.functional.upsample(value, size=(128, 128), mode='nearest')
        # print('resize后', value.shape)
        return value

    def Patch(self, intput, p=4):
        value = self.ReSIZE(intput)

        s1, s2, s3, s4 = value.shape[0], value.shape[1], value.shape[2], value.shape[3]
        value = torch.nn.functional.pixel_unshuffle(value, 4)
        s1, s2, s3, s4 = value.shape[0], value.shape[1], value.shape[2], value.shape[3]
        value = torch.reshape(value, [s1, int(s2 / 16), 16, s3, s4])
        values = value.mean(2)
        # print(values.shape)
        # print('after mean:', values.shape)
        return values

    def Cat(self, values):
        Fs = []
        Fd = []
        for index, value in values.items():
            for i, v in enumerate(value):
                if (index == 'Fs'):
                    out1 = self.Patch(v)
                    Fs.append(out1)
                elif (index == 'Fd'):
                    out2 = self.Patch(v)
                    Fd.append(out2)
        # print(type(Fs[1]))
        for i in range(len(Fs) - 1):
            if i == 0:
                Fs_out = torch.cat((Fs[i], Fs[i + 1]), dim=1)
                Fd_out = torch.cat((Fd[i], Fd[i + 1]), dim=1)
            else:
                Fs_out = torch.cat((Fs_out, Fs[i + 1]), dim=1)
                Fd_out = torch.cat((Fd_out, Fd[i + 1]), dim=1)
            # FDs = Variable(torch.from_numpy(Fs_out).float(), requires_grad=True)
            # FDg = Variable(torch.from_numpy(Fd_out).float(), requires_grad=True)
            FDs = Fs_out
            FDg = Fd_out
        return FDs, FDg


def read_mat(path):
    mat1 = cv2.imread(path, 2).astype('float32')
    mat1 = cv2.cvtColor(mat1, cv2.COLOR_GRAY2BGR)
    mat1 = mat1[None]
    tor1 = torch.from_numpy(mat1)
    tor1 = tor1.permute(0, 3, 1, 2)
    return tor1


def genRandomValue():
    x1 = torch.from_numpy(np.random.random((1, 512, 512, 64)))
    x2 = torch.from_numpy(np.random.random((1, 256, 256, 128)))
    x3 = torch.from_numpy(np.random.random((1, 128, 128, 256)))
    x4 = torch.from_numpy(np.random.random((1, 64, 64, 512)))
    x5 = torch.from_numpy(np.random.random((1, 32, 32, 512)))
    return x1, x2, x3, x4, x5


def preprocess_image(cv2im, resize_im=True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (1024, 1024))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var

# if __name__ == '__main__':
#
#     mat3 = cv2.imread('./000.png', 1)
#     mat4 = cv2.imread('./001.png', 1)
#
#     mat3 = preprocess_image(mat3)
#     mat4 = preprocess_image(mat4)
#     net = SiameseVgg19Net()
#     values = net(mat3, mat4)
#     DFF = DFF()
#     values = DFF.Cat(values)
#     FDs = values[0]
#     FDg = values[0]
