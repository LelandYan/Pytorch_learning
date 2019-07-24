# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/7/22 13:58'

import visdom
import time
import numpy as np


class Visualize:
    """
    封装了visdom的基本操作，但仍然可以通过self.vis.function或者self.function调用原生的visdom接口
    """
    def __init__(self,env="default",**kwargs):
        self.vis = visdom.Visdom(env=env,**kwargs)

        self.index = {}
        self.log_text = ""

    def reinit(self,env="default",**kwargs):
        """
        修改visdom配置
        :param env:
        :param kwargs:
        :return:
        """
        self.vis = visdom.Visdom(env=env,**kwargs)
        return self

    def plot_many(self,d):
        """
        一次plot多个
        :param d:
        :return:
        """
        for k,v in d.items():
            self.plot(k,v)

    def img_many(self,name,y,**kwargs):
        x = self.index.get(name,0)
        self.vis.line(Y=np.array([y]),X=np.array([x]),win=(name),opts=dict(title=name),update=None if x == 0 else "append",**kwargs)
        self.index[name] = x + 1

    def img(self,name,img_,**kwargs):
        self.vis.images(img_.cpu().numpy(),win=(name),opts=dict(title=name),**kwargs)

    def log(self,info,win="log_text"):
        self.log_text += ("[{time}] {info} <br>".format(time=time.strftime("%m%d_%H%M%S"),info=info))
        self.vis.text(self.log_text,win)

    def __getitem__(self, name):
        return getattr(self.vis,name)


