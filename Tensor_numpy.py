# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/21 14:41'

import torch as t
import numpy as np
from torch.autograd import Variable as V

# data = [1,2]
# tensor = t.FloatTensor(data)
# print(tensor.dot(tensor))


tensor = t.FloatTensor([[1, 2], [3, 4]])
variable = V(tensor, requires_grad=True)

t_out = t.mean(tensor * tensor)
v_out = t.mean(variable * variable)
print(t_out)
print(v_out)

v_out.backward()
print(t_out.grad)
print(v_out.grad)
print(tensor.grad)
print(variable.grad)
print(variable.data.numpy())
print(tensor.numpy())

from torch.nn import functional as F
import matplotlib.pyplot as plt

x = t.linspace(-5, 5, 200)
x = V(x)
x_np = x.data.numpy()

y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()


plt.figure(1,figsize=(8,6))
plt.subplot(221)
plt.plot(x_np,y_softplus,c="red",label="relu")
plt.ylim((-1,5))
plt.legend(loc="best")
plt.show()