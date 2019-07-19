# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/19 15:21'

import torch as t
from torch import nn
from torch.autograd import Variable as V


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.w = nn.Parameter(t.randn(in_features, out_features))
        self.b = nn.Parameter(t.randn(out_features))

    def forward(self, x):
        x = x.mm(self.w)
        return x + self.b.expand_as(x)


class Perceptron(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        nn.Module.__init__(self)
        self.layer1 = Linear(in_features, hidden_features)
        self.layer2 = Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.layer1(x)
        x = t.sigmoid(x)
        return self.layer2(x)


if __name__ == '__main__':
    percecptron = Perceptron(3, 4, 1)
    for name, param in percecptron.named_parameters():
        print(name, param.size())
