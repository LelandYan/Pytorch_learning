# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/30 17:59'
import torch
from torch.nn import functional  as F
x = torch.ones(1)
w = torch.full([1],2)
# mse = F.mse_loss(torch.ones(1),x*w)

w.requires_grad_()
mse = F.mse_loss(torch.ones(1),x*w)

# torch.autograd.grad(mse,[w])
print(torch.autograd.grad(mse,[w]))