# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 14:51:50 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import torch
import torch.nn.functional as F
#快速搭建神经网络
class basic(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(basic, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)
    
    def forward(self, x_input):
        return self.output(F.relu(self.hidden(x_input)))
net_1 = basic(1, 10, 1)

#nn.Sequential是一个有序的容器,Module会按照传入构造器的顺序依次被添加到计算图中
#https://pytorch.org/docs/stable/nn.html#torch.nn.Sequential
#https://pytorch.org/docs/stable/_modules/torch/nn/modules/container.html#Sequential
#torch.nn.Sequential(
#    torch.nn.Conv2d(1,20,5),
#    torch.nn.ReLU(),
#    torch.nn.Conv2d(20,64,5),
#    torch.nn.ReLU()
#)
#torch.nn.Sequential(torch.OrderedDict([
#    ('conv1', torch.nn.Conv2d(1,20,5)),
#    ('relu1', torch.nn.ReLU()),
#    ('conv2', torch.nn.Conv2d(20,64,5)),
#    ('relu2', torch.nn.ReLU())])
#)
net_2 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

print(net_1)
print(net_2)