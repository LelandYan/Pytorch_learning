# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/19 18:08'

import torch as t
from torch import nn
from torch.autograd import Variable as V
from torch import optim
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image


# img -> Tensor
# to_tensor = ToTensor()
# to_pil = ToPILImage()
# lena = Image.open("lena.png")
# # lena.show()
#
# # 输入是一个batch,batch_size=1
# input = to_tensor(lena).unsqueeze(0)
#
# # 锐化卷积核
# kernel = t.ones(3,3) / -9
# kernel[1][1] = 1
# conv = nn.Conv2d(1,1,(3,3),1,bias=False)
# conv.weight.data = kernel.view(1,1,3,3)
#
# out = conv(V(input))
# # to_pil(out.data.squeeze(0)).show()
#
# # 池化层是一种特殊的卷积层，用来下采样
# pool = nn.AvgPool2d(2,2)
# print(list(pool.parameters()))
# out = pool(V(input))
# # to_pil(out.data.squeeze(0)).show()
#
# input = V(t.randn(2,3))
# linear = nn.Linear(3,4)
# h = linear(input)
# print(h)
#
# bn = nn.BatchNorm1d(4)
# bn.weight.data = t.ones(4) + 4
# bn.bias.data = t.zeros(4)
#
# bn_out = bn(h)
# print(bn_out)
#
# dropout = nn.Dropout(0.5)
# o = dropout(bn_out)
# print(o)
#
# relu = nn.ReLU(inplace=True)
#
#
# net1 = nn.Sequential()
# net1.add_module("conv",nn.Conv2d(3,3,3))
# net1.add_module("batchors",nn.BatchNorm2d(3))
# net1.add_module("activation_layer",nn.ReLU())
#
# t.manual_seed(1000)
# # 输入：batch_size=3，序列长度都为2，序列中每个元素占4个维
# input = V(t.randn(2,3,4))
# # lstm输入的向量4维，3个隐藏元,1层
# lstm = nn.LSTM(4,3,1)
# # 初始状态：1层，batch_size=3,3个单元
# h0 = V(t.randn(1,3,3))
# c0 = V(t.randn(1,3,3))
# out,hn = lstm(input,(h0,c0))
# print(out)

# score = V(t.randn(3,2))
# label = V(t.Tensor([1,0,1])).long()
# criterion = nn.CrossEntropyLoss()
# loss = criterion(score,label)
# print(loss)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,16*5*5)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    pass
    # net = Net()
    # optimizer = optim.SGD(params=net.parameters(),lr=1)
    # optimizer.zero_grad()
    #
    # input = V(t.randn(1,3,32,32))
    # output = net(input)
    # output.backward()
    #
    # # 执行优化
    # optimizer.step()
    #
    # # 调整学习率的两种方法
    # optimizer = optim.SGD([
    #     {"params":net.features.parameters()},
    #     {"params":net.classifier.parameters(),"lr":1e-2}
    # ],lr=1e-5)
    #
    #
    #
    # special_layers = nn.ModuleList([net.classifier[0],net.classifier[3]])
    # special_layers_params = list(map(id,special_layers.parameters()))
    # base_params = filter(lambda p:id(p) not in special_layers_params,net.parameters())
    # optimizer = t.optim.SGD([
    #     {"params":base_params},
    #     {"params":special_layers.parameters(),"lr":0.01}
    # ],lr=0.001)

    input = V(t.rand(2,3))
    model = nn.Linear(3,4)
    output1 = model(input)
    output2 = nn.functional.linear(input,model.weight,model.bias)
    print(output1==output2)

    b = nn.functional.relu(input)
    b2 = nn.ReLU(input)
    print(b == b2)
