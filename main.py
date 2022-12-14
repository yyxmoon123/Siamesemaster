import cv2
import torch
import numpy as np
import time

import argparse

from torch.utils.data import DataLoader, Dataset
from dataset import DRGdataset

from models.vgg import SiameseVgg19Net, DFF
from models.AE import AE
import models.AEloss as AELoss

parser = argparse.ArgumentParser(description='SiameseNet')
parser.add_argument('--input', action='store', default='./data', help='')
parser.add_argument('--number_work', action='store', type=int, default=0, help='')
parser.add_argument('--save_model', action='store', default='./weight', help='')
parser.add_argument('--epoch', action='store', type=int, default=30, help='eopch')
parser.add_argument('--gpu', action='store', default='cuda:1', help='')
parser.add_argument('--batch_size', action='store', type=int, default=1, help='')
parser.add_argument('--cpuswitch', action='store', type=bool, default=False)
parser.add_argument('--continues', action='store', type=bool, default=False)
parser.add_argument('--begin', action='store', type=int, default=0, help='')
parser.add_argument('--weight', action='store', default='')
opt = parser.parse_args()
print(opt)

train_dataset = DRGdataset()

train_dataloder = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, num_workers=opt.number_work,
                             shuffle=False)
if opt.cpuswitch:
    device = torch.device('cpu')
else:
    device = torch.device(opt.gpu if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    SiameseVgg19Net = SiameseVgg19Net().to(device)
    ae = AE().to(device)
    if opt.continues:
        print(opt.weight)
        ae = torch.load(opt.weight)
    dff = DFF()
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
    lall = []
    i = 0
    loss = 0
    ep = 0
    for epoch in range(opt.epoch):
        print(epoch)
        ep = opt.begin + epoch
        start = time.clock()
        for index, data in enumerate(train_dataloder):
            Is = data[0].to(device)
            Ig = data[1].to(device)
            value = SiameseVgg19Net(Is, Ig)
            value = dff.Cat(value)
            out = ae(value[0], value[1])
            L_all = AELoss.L_all(out[0], out[1], value[0])
            loss = L_all.item()
            optimizer.zero_grad()  # ???????????????????????????????????????
            L_all.backward()  # ???????????????????????????????????????, ?????????????????????
            optimizer.step()
        end = time.clock()
        print('epoch:' + str(ep) + 'loss :' + str(loss) + 'runtime:' + str(end - start))
        with open('log.txt', 'w') as F:
            str1 = 'run: ' + str(ep) + 'loss: ' + str(loss) + 'time:' + str(end - start)
            F.write(str1)
        i = ep + 1
        if i % 10 == 0:
            torch.save(ae, './weight/ad%d.pt' % i)
torch.save(ae, './weight/ae%d.pt' % i)
