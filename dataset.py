import cv2
from torch.utils.data import DataLoader, Dataset
import os
import glob
import numpy as np
import torch
from PIL import Image

from torchvision.transforms import transforms as Transforms
from tkinter import _flatten


class DRG():
    def __init__(self):
        self.center = (np.random.randint(0, 700), np.random.randint(0, 700))
        self.axes = (np.random.randint(0, 30), np.random.randint(0, 75))
        self.angle = np.random.randint(0, 360)

    def drg(self, mat):
        mat = cv2.resize(mat, (720, 720), interpolation=cv2.INTER_NEAREST)
        blank_image = np.zeros((720, 720, 3), np.uint8)
        blank_image.fill(0)

        cv2.ellipse(img=blank_image, center=self.center, axes=self.axes, angle=self.angle, startAngle=0, endAngle=360,
                    color=(21, 42, 31), thickness=-1)
        blank_image = cv2.bitwise_or(mat, blank_image)
        return blank_image


def getPath():
    img_path = []
    path = './data/mvtec'
    paths = os.listdir(path)
    for i in paths:
        if os.path.isdir(os.path.join(path, i)):
            subdirs = os.listdir(os.path.join(path, i))
            for _, j in enumerate(subdirs):
                if subdirs[_] == 'train':
                    dirs = os.path.join(path, i, j)
                    files = glob.glob(os.path.join(dirs, 'good', '*.png'))
                    img_path.append(files)
                    img_path = list(_flatten(img_path))
    return img_path


transform = Transforms.Compose([
    Transforms.Resize([720, 720]),
    Transforms.ToTensor(),
    Transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class DRGdataset(Dataset):
    def __init__(self):
        self.imgsPath = getPath()
        self.DRG = DRG()

    def __getitem__(self, item):
        img = cv2.imread(self.imgsPath[item], 1)
        mask = self.DRG.drg(mat=img)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = transform(img=img)
        mask = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        mask = transform(img=mask)
        return img, mask

    def __len__(self):
        return len(self.imgsPath)


#
if __name__ == '__main__':
    path = './models/000.png'
    DRGdataset = DRGdataset()
    print(DRGdataset[0][1])
    img = cv2.imread(path, 1)
