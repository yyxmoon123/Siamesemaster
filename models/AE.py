import torch.nn as nn
import torch.nn.functional as F


# def conv1x1(in_planes, out_planes):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1)
#
#
# class BasicBlock(nn.Module):
#     def __init__(self, inchannel, outchannel):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv1x1(inchannel, outchannel)
#         self.bn1 = nn.BatchNorm2d(outchannel)
#         ###


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


