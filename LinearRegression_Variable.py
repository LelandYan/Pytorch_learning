# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/19 14:53'

import torch as t
from torch.autograd import Variable as V
import matplotlib.pyplot as plt


# 设置随机种子，保证在不同计算机上的运行时间相同
t.manual_seed(1000)

def get_fake_data(batch_size=8):
    x = t.rand(batch_size,1) * 20
    y = x * 2 + (1 + t.randn(batch_size,1)) * 3
    return x,y

if __name__ == '__main__':
    w = V(t.rand(1,1),requires_grad=True)
    b = V(t.zeros(1,1),requires_grad=True)

    lr = 0.001
    for ii in range(8000):
        x,y = get_fake_data()
        x,y = V(x),V(y)

        y_pred = x.mm(w) + b.expand_as(y)
        loss = 0.5 * (y_pred - y) ** 2
        loss = loss.sum()

        loss.backward()

        w.data.sub_(lr * w.grad.data)
        b.data.sub_(lr * b.grad.data)

        w.grad.data.zero_()
        b.grad.data.zero_()

        if ii % 1000 == 0:
            plt.clf()

            x = t.arange(0,20).view(-1,1).float()
            y = x.mm(w.data) + b.data.expand_as(x)
            plt.plot(x.numpy(),y.numpy())

            x2,y2 = get_fake_data(batch_size=20)
            plt.scatter(x2.numpy(),y2.numpy())

            plt.xlim(0,20)
            plt.ylim(0,41)
            plt.pause(0.5)
    print(w.data.squeeze().item(),b.data.squeeze().item())