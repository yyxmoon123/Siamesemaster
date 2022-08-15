import time

import torch
import torchvision
import numpy as np
import os
import torch.nn as nn
import torchvision.models as models
from dataset import DRGdataset
from torch.utils.data import DataLoader


class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True)
        for i,m in enumerate(self.modules()):
            print(i,m,type(m))

    def forward(self, x):
        x1 = self.branch(x)
        x2 = self.branch(x)
        return x1, x2

    def branch(self, allin):
        x = self.vgg19(allin)

        return x


if __name__ == '__main__':

    start = time.clock()

    SiameseNet = SiameseNet()
    DRGdataset = DRGdataset()
    te_dataloder = DataLoader(dataset=DRGdataset, batch_size=1, shuffle=False, num_workers=2)
    # for index, data in enumerate(te_dataloder):
    #     print(index)
    end = time.clock()
    print(end-start)
