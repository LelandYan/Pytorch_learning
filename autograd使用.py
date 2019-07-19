# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/19 14:42'

import torch as t
from torch.autograd import Function

class Mul(Function):
    @staticmethod
    def forward(ctx,w,x,b,x_requires_grad=True):
        ctx.x_requires_grad = x_requires_grad
        ctx.save_for_backward(w,x)
        output = w * x + b
        return output

    @staticmethod
    def backward(ctx,grad_output):
        w,x = ctx.saved_variables
        grad_w = grad_output * x
        if ctx.x_requires_grad:
            grad_x = grad_output * w
        else:
            grad_x = None
        grad_b = grad_output * 1
        return grad_w,grad_x,grad_b,None