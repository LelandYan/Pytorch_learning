# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/22 12:29'

import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T


class DogCat(data.Dataset):
    def __init__(self,root,transforms=None,train=True,test=False):
        """
        目标：获取所有图片地址，并根据训练、验证、测试划分数据
        """
        self.test = test
        imgs = [os.path.join(root,img) for img in os.listdir(root)]
        if self.test:
            imgs = sorted(imgs,key=lambda x:int(x.split(".")[-2].split("\\")[-1]))
        else:
            imgs = sorted(imgs,key=lambda x:int(x.split(".")[-2]))
        imgs_num = len(imgs)

        # 划分训练、验证集，验证:训练=3:7
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7*imgs_num)]
        else:
            self.imgs = imgs[int(0.7*imgs_num):]

        if transforms is None:
            # 数据转换操作，测试验证和训练的数据转换有所区别
            normalize = T.Normalize(mean=[0.485,0.456,0.406],
                                    std=[0.229,224,0.225])
            # 测试集和验证集
            if self.test or not train:
                self.transform = T.Compose([
                    T.Scale(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            # 训练集
            else:
                self.transform = T.Compose([
                    T.Scale(256),
                    T.RandomSizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        """
        返回一张图片的数据
        如果是测试集，没有图片id
        """
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split(".")[-2].split("\\")[-1])
        else:
            label = 1 if "dog" in img_path.split("\\")[-1] else 0
        data = Image.open(img_path)
        data = self.transform(data)
        return data,label

    def __len__(self):
        """
        返回数据中的所有图片的个数
        """
        return len(self.imgs)
