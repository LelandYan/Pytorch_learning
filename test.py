# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/18 12:18'

import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        # nn.Module 子类的函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()
        # 卷积层1表示输入图片为单通道，6表示输出通道树，5表示卷积核为5*5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 卷积层
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 仿射层/全连接层，y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积 -> 激活  ->  池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2))
        # reshape -1 表示自适应
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def f(x):
    y = x ** 2 * t.exp(x)
    return y

def gradf(x):
    """手动的求导"""
    dx = 2*x*t.exp(x) + x ** 2 * t.exp(x)
    return dx
def abs(x):
    if x.data[0]>0:return x
if __name__ == '__main__':
    pass
    # net = Net()
    # # print(net)
    # # params = list(net.parameters())
    # # print(len(params))
    # # for name, parameters in net.named_parameters():
    # #     print(name, ":", parameters.size())
    # input = Variable(t.randn(1,1,32,32))
    # # out = net(input)
    # # print(out.size())
    # # net.zero_grad()
    # # out.backward(Variable(t.ones(1,10)))
    # output = net(input)
    # target = Variable(t.arange(0,10)).float()
    # criterion = nn.MSELoss()
    # loss = criterion(output,target)
    # print(loss)
    #
    # # 所有的参数梯度清零
    # net.zero_grad()
    # print("反向传播之前conv1.bias的梯度")
    # print(net.conv1.bias.grad)
    # loss.backward()
    # print("反向传播之后cov1.bias的梯度")
    # print(net.conv1.bias.grad)
    #
    # learning_rate = 0.01
    # for f in net.parameters():
    #     f.data.sub_(f.grad.data * learning_rate)
    #
    # optimizer = optim.SGD(net.parameters(),lr=0.01)
    # optimizer.zero_grad()
    #
    # output = net(input)
    # loss = criterion(output,target)
    #
    # loss.backward()
    # optimizer.step()
    # a = t.Tensor(2,3)
    # print(a)
    # b = t.Tensor([[1,2,3],[4,5,6]])
    # print(b)
    # print(b.tolist())
    # b_size = b.size()
    # print(b_size)
    # print(b.shape)
    # print(t.ones(2,3))
    # print(t.zeros(2,3))
    # print(t.arange(1,6,2))
    # print(t.linspace(1,10,3))
    # print(t.randn(2,3))
    # print(t.randperm(5))
    # print(t.eye(2,3))
    # a = t.arange(0,6)
    # print(a.view(2,3))
    # b = a.view(-1,3)
    # print(b)
    # print(b.unsqueeze(1))
    # print(b.unsqueeze(-2))
    # c = b.view(1,1,1,2,3)
    # print(c)
    # print(c.squeeze(0))
    # a = t.randn(3,4)
    # print(a)
    # print(a[0])
    # a = t.arange(0,16).view(4,4)
    # index = t.LongTensor([[0,1,2,3]])
    # print(a.gather(0,index))
    # index = t.LongTensor([[3,2,1,0]]).t()
    # print(index)
    # a = t.ones(3,2)
    # b = t.zeros(2,3,1)
    # print(t.rand(4,1))
    # a = V(t.ones(3,4),requires_grad=True)
    # print(a)
    # b = V(t.zeros(3,4))
    # print(b)
    # c = a.add(b)
    # print(c)
    # d = c.sum()
    # # 进行反向传播
    # d.backward()
    # print(c.data.sum(),c.sum())
    # print(a.grad)
    # print(a.required_grad,b.requires_grad,c.requires_grad)
    # x = V(t.randn(3,4),requires_grad=True)
    # y = f(x)
    # print(y)
    # y.backward(t.ones(y.size()))
    # print(x.grad)
    # print(gradf(x))
    # x = V(t.ones(1))
    # b = V(t.rand(1),requires_grad=True)
    # w = V(t.rand(1),requires_grad=True)
    # y = w * x
    # z = y + b
    # print(x.requires_grad,b.requires_grad,w.requires_grad)
    # print(x.is_leaf,w.is_leaf,b.is_leaf)
    # print(y.is_leaf,z.is_leaf)
    x = V(t.ones(1),requires_grad=True)
    y = abs(x)
    y.backward()
    print(x.grad)