# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/30 17:59'
import torch
from torch.nn import functional  as F
# x = torch.ones(1)
# w = torch.full([1],2)
# # mse = F.mse_loss(torch.ones(1),x*w)
#
# w.requires_grad_()
# mse = F.mse_loss(torch.ones(1),x*w)
#
# # torch.autograd.grad(mse,[w])
# print(torch.autograd.grad(mse,[w]))
#


# a = torch.rand(3)
# a.requires_grad_()
#
# # p = F.softmax(a,dim=0)
# # p.backward()
#
# p = F.softmax(a,dim=0)
# print(p)
# print(torch.autograd.grad(p[1],[a],retain_graph=True))
# print(torch.autograd.grad(p[2],[a]))


# x = torch.randn(1,10)
# w = torch.randn(1,10,requires_grad=True)
#
# o = torch.sigmoid(x@w.t())
#
# loss = F.mse_loss(torch.ones(1,1),o)
# loss.backward()
#
# print(w.grad)
#
# x = torch.randn(1,10)
# w = torch.randn(2,10,requires_grad=True)
#
# o = torch.sigmoid(x@w.t())
#
# loss = F.mse_loss(torch.ones(1,2),o)
# loss.backward()
#
# print(w.grad)
#
# x = torch.tensor([0,0],requires_grad=True)
# optimizer = torch.optim.Adam([x],lr=1e-3)


from torch.utils.data import random_split
