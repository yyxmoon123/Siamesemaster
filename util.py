import sys

import cv2
import os
import PIL
import numpy as np

path = './testimg/1.png'


class DefectRandomGeneration():
    def __init__(self, input, SL, SH, BL, BH, R1, R2):
        if not os.path.isfile(input):
            raise Exception('input is not file ,maybe dir?')
        self.input = input
        self.Sl = SL
        self.Sh = SH
        self.Bl = BL
        self.Bh = BH
        self.R1 = R1
        self.R2 = R2
        pass

    def drg(self):
        W = self.input.shape[1]
        H = self.input.shape[0]
        Xr = 1
        return


def noise(img, snr, h1, w1, center):
    h = h1
    w = w1
    img1 = img.copy()
    sp = h * w  # 计算图像像素点个数
    NP = int(sp * (1 - snr))  # 计算图像椒盐噪声点个数
    for i in range(NP):
        randx = np.random.randint(center[0] - h - 1, center[0] + h - 1)  # 生成一个 1 至 h-1 之间的随机整数
        randy = np.random.randint(center[1] - w - 1, center[1] + w - 1)  # 生成一个 1 至 w-1 之间的随机整数
        if np.random.random() <= 0.5:  # np.random.random()生成一个 0 至 1 之间的浮点数
            img1[randx, randy] = 0
        else:
            img1[randx, randy] = 255
    return img1


def paintBlock(c1=1, c2=1, r1=None, r2=None, input=None, pi=None):
    H = input.shape[0]
    W = input.shape[1]
    centenX = np.random.randint(0, 255)
    centerY = np.random.randint(0, 255)
    center = (centenX, centerY)
    X = np.random.randint(H * r1, H * r2)
    Y = np.random.randint(W * r1, W * r2)
    # cv2.ellipse(input, center, (X, Y), 0, 0, 360, (0, 255, 0))
    snr = np.random.random()
    mat1 = noise(input, snr, X, Y, center)
    return mat1


if __name__ == "__main__":
    mat = cv2.imread(filename='./data/Yellow0.png', flags=2)

    for i in range(100):
        mat1 = paintBlock(r1=0.04, r2=0.06, input=mat, pi=4)
        cv2.imwrite(os.path.join('./noise', str(i) + '.jpg'), mat1)
        sys.stdout.write(
            ' \routput %i 图' % i
        )
