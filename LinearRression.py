# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/24 21:57'

import numpy as np
import torch


def compute_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current,w_current,points,learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient += -(2 / N) * (y - ((w_current * x ) + b_current))
        w_gradient += -(2 / N) * x * (y - ((w_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b,new_w]

def gradient_descent_runner(points,starting_b,starting_w,learning_rate,num_iterations):
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        b,w = step_gradient(b,w,np.array(points),learning_rate)
    return [b,w]

import torch
# import numpy as np
# print(torch.ones((2,2)))

a = np.random.randn(4, 3) # a.shape = (4, 3)
b = np.random.randn(3, 2) # b.shape = (3, 2)
c = a*b
print(c)


# a = [[ 0.67826139] [ 0.29380381] [ 0.90714982] [ 0.52835647] [ 0.4215251 ] [ 0.45017551] [ 0.92814219] [ 0.96677647] [ 0.85304703] [ 0.52351845] [ 0.19981397] [ 0.27417313] [ 0.60659855] [ 0.00533165] [ 0.10820313] [ 0.49978937] [ 0.34144279] [ 0.94630077]]

