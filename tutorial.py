# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/21 15:16'

import torch as t
from torch.autograd import Variable as V
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn

# x = t.unsqueeze(t.linspace(-1, 1, 100), dim=1)
# y = x.pow(2) + 0.2 * t.rand(x.size())
n_data = t.ones(100,2)
x0 = t.normal(2*n_data,1)
y0 = t.zeros(100)
x1 = t.normal(-2*n_data,1)
y1 = t.ones(100)
x = t.cat((x0,x1),0).type(t.FloatTensor)
y = t.cat((y0,y1),0).type(t.LongTensor)



x, y = V(x), V(y)


# plt.scatter(x[:,0],x[:,1],c=y,s=100,lw=0)
# plt.show()

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(2, 10, 2)
plt.ion()
plt.show()

optimizer = t.optim.SGD(net.parameters(), lr=0.02)
loss_func = t.nn.CrossEntropyLoss()

for i in range(100):
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 2 == 0:
        plt.cla()
        # plt.scatter(x, y)
        # plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        # plt.text(0.5, 0, f"Loss={loss.item()}", fontdict={"size": 20, "color": "red"})
        # plt.pause(0.1)
        prediction = t.max(F.softmax(prediction),1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x[:,0],x[:,1],c=pred_y,s=100,lw=0)
        accuracy = sum(pred_y == target_y) / 200
        plt.text(1.5, -4, f"Accuracy={accuracy}", fontdict={"size": 20, "color": "red"})
        plt.pause(1)

plt.ioff()
plt.show()
