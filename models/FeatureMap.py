import os

import cv2
import torch
import numpy as np
from torch.autograd import Variable
from vgg import SiameseVgg19Net
from AE import AE
import PIL


def preprocess_image(cv2im, resize_im=True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
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


# def get_feature(self):
#     # input = Variable(torch.randn(1, 3, 224, 224))
#     input = self.process_image()
#     print("input shape", input.shape)
#     x = input
#     for index, layer in enumerate(self.pretrained_model):
#         # print(index)
#         # print(layer)
#         x = layer(x)
#         if (index == self.selected_layer):
#             return x
#
#
# def get_single_feature(self):
#     features = self.get_feature()
#     print("features.shape", features.shape)
#     feature = features[:, 0, :, :]
#     print(feature.shape)
#     feature = feature.view(feature.shape[1], feature.shape[2])
#     print(feature.shape)
#     return features
#
#
# def save_feature_to_img(self):
#     # to numpy
#     features = self.get_single_feature()
#     for i in range(features.shape[1]):
#         feature = features[:, i, :, :]
#         feature = feature.view(feature.shape[1], feature.shape[2])
#         feature = feature.data.numpy()
#         # use sigmod to [0,1]
#         feature = 1.0 / (1 + np.exp(-1 * feature))
#         # to [0,255]
#         feature = np.round(feature * 255)
#         print(feature[0])
#         os.mkdir('./feature/' + str(self.selected_layer))
#         cv2.imwrite('./feature/' + str(self.selected_layer) + '/' + str(i) + '.jpg', feature)


def get_single_feature1(x):
    features = x
    print("features.shape", features.shape)
    feature = features[:, 0, :, :]
    print(feature.shape)
    feature = feature.view(feature.shape[1], feature.shape[2])
    print(feature.shape)
    return features


def save_feature_to_img1(x):
    # to numpy
    features = x
    for i in range(features.shape[1]):
        feature = features[:, i, :, :]
        feature = feature.view(feature.shape[1], feature.shape[2])
        feature = feature.data.cpu().numpy()
        # use sigmod to [0,1]
        feature = 1.0 / (1 + np.exp(-1 * feature))
        # to [0,255]
        feature = np.round(feature * 255)
        print(feature[0])

        cv2.imwrite('./feature/' + str(i) + '.jpg', feature)


def show_heatmap(feature, output_jpg_name, row_image):
    data = np.squeeze(feature,axis=0)
    size = row_image.shape
    heatmap = data.sum(0) / data.shape[0]
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = 1.0 - heatmap  # 也可以不写，就是蓝色红色互换的作用
    heatmap = cv2.resize(heatmap, (size[0], size[1]))  # (224,224)指的是图像的size，需要resize到原图大小
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    row = row_image
    # row = row.transpose(1, 2, 0)
    superimposed_img = heatmap * 1.0 + row * 0.5  # 1.0 和 0.5代表heatmap和row image的强度占比，可调整
    out1 = np.hstack([row,superimposed_img,heatmap])
    cv2.imwrite(output_jpg_name, out1)


if __name__ == '__main__':
    img1 = cv2.imread('./000.png', 1)
    img2 = cv2.imread('./001.png', 1)
    mat1 = preprocess_image(img1).cuda()
    mat2 = preprocess_image(img2).cuda()
    SiameseVgg19Net = SiameseVgg19Net().cuda()
    SiameseVgg19Net.eval()
    value = SiameseVgg19Net(mat1, mat2)
    Fs, Fd = value['Fs'], value['Fd']

    AE = AE().cuda()
    AE.eval()

    x = Fs[15]
    n = get_single_feature1(x=x)
    n = save_feature_to_img1(x=n)
    x = x.detach().cpu().numpy()
    show_heatmap(x,'./test1.jpg',img1)
