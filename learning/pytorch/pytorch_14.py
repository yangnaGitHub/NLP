# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 11:15:13 2019

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import torch
import matplotlib.pyplot as plt

N_SAMPLES = 20
N_HIDDEN = 300

x_data = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)#(20,)=>(20, 1)
y_data = x_data + 0.3*torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))#均值,标准差

t_x_data = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)#(20,)=>(20, 1)
t_y_data = t_x_data + 0.3*torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))#均值,标准差

plt.scatter(x_data.data.numpy(), y_data.data.numpy(), c='red', s=50, alpha=0.5, label='train')
plt.scatter(t_x_data.data.numpy(), t_y_data.data.numpy(), c='blue', s=50, alpha=0.5, label='test')
plt.legend(loc='upper left')
plt.ylim(-2.5, 2.5)
plt.show()

net_ov = torch.nn.Sequential(
        torch.nn.Linear(1, N_HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(N_HIDDEN, N_HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(N_HIDDEN, 1))

net_dropout = torch.nn.Sequential(
        torch.nn.Linear(1, N_HIDDEN),
        torch.nn.Dropout(0.5),
        torch.nn.ReLU(),
        torch.nn.Linear(N_HIDDEN, N_HIDDEN),
        torch.nn.Dropout(0.5),
        torch.nn.ReLU(),
        torch.nn.Linear(N_HIDDEN, 1))

#print(net_ov)
#print(net_dropout)

opt_ov = torch.optim.Adam(net_ov.parameters(), lr=0.01)
opt_dropout = torch.optim.Adam(net_dropout.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()

for step in range(500):
    pred_ov = net_ov(x_data)
    loss_ov = loss_func(pred_ov, y_data)
    opt_ov.zero_grad()
    loss_ov.backward()
    opt_ov.step()
    
    pred_dropout = net_dropout(x_data)
    loss_dropout = loss_func(pred_dropout, y_data)
    opt_dropout.zero_grad()
    loss_dropout.backward()
    opt_dropout.step()
    
    if 0 == step % 10:
        net_ov.eval()
        net_dropout.eval()#检测[预测]的时候要将dropout的功效取消掉
        
        plt.cla()
        t_pred_ov = net_ov(t_x_data)
        t_pred_dropout = net_dropout(t_x_data)
        plt.scatter(x_data.data.numpy(), y_data.data.numpy(), c='red', s=50, alpha=0.5, label='train')
        plt.scatter(t_x_data.data.numpy(), t_y_data.data.numpy(), c='magenta', s=50, alpha=0.5, label='test')
        plt.plot(t_x_data.data.numpy(), t_pred_ov.data.numpy(), 'r-', lw=3, label='overfitting')
        plt.plot(t_x_data.data.numpy(), t_pred_dropout.data.numpy(), 'b--', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, 'overfitting loss=%.4f' % loss_func(t_pred_ov, t_y_data).data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.text(0, -1.5, 'dropout loss=%.4f' % loss_func(t_pred_dropout, t_y_data).data.numpy(), fontdict={'size': 20, 'color':  'blue'})
        plt.legend(loc='upper left')
        plt.ylim((-2.5, 2.5))
        plt.pause(0.1)
        
        net_ov.train()
        net_dropout.train()#训练的时候要将的dropout打开

plt.ioff()
plt.show()

数据之间的规律,全连接层和激励函数之间