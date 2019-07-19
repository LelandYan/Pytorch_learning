# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/19 18:08'

import torch as t
from torch import nn
from torch.autograd import Variable as V
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

# img -> Tensor
to_tensor = ToTensor()
to_pil = ToPILImage()
lena = Image.open("lena.png")
# lena.show()

# 输入是一个batch,batch_size=1
input = to_tensor(lena).unsqueeze(0)

# 锐化卷积核
kernel = t.ones(3,3) / -9
kernel[1][1] = 1
conv = nn.Conv2d(1,1,(3,3),1,bias=False)
conv.weight.data = kernel.view(1,1,3,3)

out = conv(V(input))
# to_pil(out.data.squeeze(0)).show()

# 池化层是一种特殊的卷积层，用来下采样
pool = nn.AvgPool2d(2,2)
print(list(pool.parameters()))
out = pool(V(input))
# to_pil(out.data.squeeze(0)).show()

input = V(t.randn(2,3))
linear = nn.Linear(3,4)
h = linear(input)
print(h)

bn = nn.BatchNorm1d(4)
bn.weight.data = t.ones(4) + 4
bn.bias.data = t.zeros(4)

bn_out = bn(h)
