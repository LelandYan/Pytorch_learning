# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/18 19:42'
import torch as t
import numpy as np
import torch.nn as nn
import torchvision as tv
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

# 可以把Tensor转化为Image,方便可视化
show = ToPILImage()
classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
# 定义对数据的预处理
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 训练集
trainset = tv.datasets.CIFAR10(
    root="/data/",
    train=True,
    download=True,
    transform=transforms
)
trainloader = t.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

# 测试集
testset = tv.datasets.CIFAR10(
    root="/data/",
    train=False,
    download=True,
    transform=transforms
)

testloader = t.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)


class Net(nn.Module):
    def __init__(self):
        # nn.Module 子类的函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()
        # 卷积层3表示输入图片为单通道，6表示输出通道树，5表示卷积核为5*5
        self.conv1 = nn.Conv2d(3, 6, 5)
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

if __name__ == '__main__':
    # (data, label) = trainset[100]
    # print(classes[label])
    # # show((data + 1) / 2).resize((100, 100))
    # img1 = ToPILImage()((data + 1) / 2)
    # img1.show()
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    # print(" ".join("%11s" % classes[labels[j]] for j in range(4)))
    # img2 = ToPILImage()(tv.utils.make_grid((images + 1) / 2))
    # img2.show()

    # 模型创建
    net = Net()
    # 定义损失函数，使用交叉熵
    criterion = nn.CrossEntropyLoss()
    # 定义优化器，使用SGD随机梯度下降法
    optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
    # 进行模型的训练
    for epoch in range(2):
        running_loss = 0.0
        for i,data in enumerate(trainloader,0):
            # 输入数据
            inputs,labels = data
            inputs,labels = Variable(inputs),Variable(labels)

            # 梯度清零
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()

            # 更新参数
            optimizer.step()

            # 打印log信息
            running_loss += loss.item()

            if i % 2000 == 1999:
                print(f"{epoch+1},{i+1} loss: {running_loss/2000}")
                running_loss = 0

    print("Finished Training")
    # 模型的检验
    dataiter = iter(testloader)
    images,labels = dataiter.next()
    print("实际的label："," ".join("%11s" % classes[labels[j]] for j in range(4)))
    # img2 = ToPILImage()(tv.utils.make_grid((images / 2 ) - 0.5))
    # img2.show()
    outputs = net(Variable(images))
    _,predicted = t.max(outputs.data,1)
    print("预测结果：", " ".join("%5s" % classes[predicted[j]] for j in range(4)))

    # 在整个测试集上的效果
    correct = 0
    total = 0
    for data in testloader:
        images,labels = data
        outputs = net(Variable(images))
        _,predicted = t.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print(f"10000张测试集中的准确率为：{ 100 * correct / total}")