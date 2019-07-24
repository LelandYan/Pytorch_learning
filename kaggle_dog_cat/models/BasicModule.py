# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/22 13:35'
from torch import nn
import torch as t
import time


class BasicModule(nn.Module):
    """
    封装了nn.Module,主要提供save和load两个方法
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))  # 模型默认的名字

    def load(self, path):
        """
        可加载指定路径的模型
        """
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        缓存模型，默认使用模型名字+时间作为文件名
        """
        if name is None:
            prefix = "checkpoints/" + self.model_name + "_"
            name = time.strftime(prefix + "%m%d_%H:%M:%S.pth")
        t.save(self.state_dict(), name)
        return name
