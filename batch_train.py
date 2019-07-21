# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/21 16:10'

import torch
import torch.utils.data as Data

BATCH_SIZE = 8

x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)

torch_dataset = Data.TensorDataset(x,y)
loader = Data.DataLoader(dataset=torch_dataset,shuffle=True,batch_size=BATCH_SIZE,num_workers=2)
if __name__ == '__main__':
    pass
    for epoch in range(3):
        for step,(batch_x,batch_y) in enumerate(loader):
            print("Epoch: ",epoch,"| Step: ",step,"| batch x: ",batch_x.numpy(),"| batch_y: ",batch_y.numpy())
