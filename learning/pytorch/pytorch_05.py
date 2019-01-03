# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 14:52:29 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import torch
from torch.autograd import Variable

torch.manual_seed(1)
x_data = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y_data = x_data.pow(2) + 0.2*torch.rand(x_data.size())
x_val, y_val = Variable(x_data), Variable(y_data)

#保存网络
def save():
    net_1 = torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1))
    optimizer = torch.optim.SGD(net_1.paramaters(), lr=0.5)
    loss_func = torch.nn.MSELoss()
    
    for times in range(100):
        prediction = net_1(x_val)
        loss = loss_func(prediction, y_val)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #保存整个网络
    torch.save(net_1, '05.pkl')
    #保存网络中的参数(速度快,占内存少)
    torch.save(net_1.state_dict(), '05_params.pkl')

#restore
def restore_net():
    net_2 = torch.load('05.pkl')
    prediction = net_2(x_val)
    
def restore_params():
    net_3 = torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1))
    net_3.load_state_dict(torch.load('05_params.pkl'))
    prediction = net_3(x_val)