# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/21 11:29'

import torch as t
from torch.utils import data
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

transform = T.Compose([
    T.Scale(224),# 调整图片尺寸，长宽比保持不变,最短边为224像素
    T.CenterCrop(224),# 从图片中间切出224*224的图片
    T.ToTensor(),#将图片（Image）转成Tensor，归一化到【0,1】
    T.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
])


class DogCat(data.Dataset):
    def __init__(self, root,transform=None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transform = transform

    def __getitem__(self, item):
        img_path = self.imgs[item]
        label = 1 if "dog" in img_path.split("\\")[-1] else 0
        data = Image.open(img_path)
        if self.transform:
            data  = self.transform(data)
        return data,label

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    pass
    # datasets = DogCat(r"C:\Users\lenovo\Desktop\Pytorch_learning\data\dogcat",transform)
    # img,label = datasets[0]
    # for img,label in datasets:
    #     print(img.size(),img.float().mean(),label)
    datasets = ImageFolder(r"data\\dogcat\\")
    # print(datasets.class_to_idx)
    # print(datasets.imgs)
    # print(datasets[0][0])
    dataload = data.DataLoader(datasets,batch_size=3,shuffle=True,num_workers=0,drop_last=False)
    dataiter = iter(dataload)
    for batch_datas,batch_labels in dataiter:
        print(batch_datas)