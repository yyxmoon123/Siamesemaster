import cv2
import torch
import numpy as np

import argparse

from torch.utils.data import DataLoader, Dataset
from dataset import DRGdataset

from models.vgg import SiameseVgg19Net, DFF
from models.AE import AE
import models.AEloss as AELoss

parser = argparse.ArgumentParser(description='SiameseNet')
parser.add_argument('--input', action='store', default='./data', help='')
parser.add_argument('--numberwork', action='store', default=0, help='')
parser.add_argument('--save_model', action='store', default='./weight', help='')
parser.add_argument('--epoch', action='store', default=60, help='eopch')
parser.add_argument('--gpu',action='store',default='cuda:1',help='')
opt = parser.parse_args()
print(opt)

train_dataset = DRGdataset()

train_dataloder = DataLoader(dataset=train_dataset, batch_size=1, num_workers=opt.numberwork)
device = torch.device( 'cpu')

if __name__ == '__main__':
    SiameseVgg19Net = SiameseVgg19Net().to(device)
    ae = AE().to(device)
    dff = DFF()
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
    lall = []
    i = 0
    loss = 0
    with open('log.txt',"w+") as F:
        for epoch in range(opt.epoch):
            for index, data in enumerate(train_dataloder):
                Is = data[0].to(device)
                Ig = data[1].to(device)
                value = SiameseVgg19Net(Is, Ig)
                value = dff.Cat(value)

                out = ae(value[0], value[1])
                L_all = AELoss.L_all(out[0], out[1], value[0])
                loss = L_all.item()
                optimizer.zero_grad()  # 清空上一步的残余更新参数值
                L_all.backward()  # 以训练集的误差进行反向传播, 计算参数更新值
                optimizer.step()
            print('epoch:'+str(epoch)+'loss :'+str(loss))
            F.write(str(loss))
            i = i + 1
    torch.save(ae, './weight/ae%d.pt' % i)
