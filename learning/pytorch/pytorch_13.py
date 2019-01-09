# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 11:11:01 2019

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)
np.random.seed(1)

BATCH_SIZE = 64
ART_COMPONENTS = 15
paint_points = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])#1*15 ==> 64*15

def true_result():
    t_data = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]#64*1,[1,2]中间的均匀分布
    paintings = t_data * np.power(paint_points, 2) + (t_data - 1)#np.power(paint_points, 2) => 1-0-1
    return torch.from_numpy(paintings).float()

N_DATA = 5
ART_COMPONENTS = 15
Generator = nn.Sequential(
        nn.Linear(N_DATA, 128),
        nn.ReLU(),
        nn.Linear(128, ART_COMPONENTS))

Discriminator = nn.Sequential(
        nn.Linear(ART_COMPONENTS, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid())

lr_G = 0.0001
lr_D = 0.0001
opt_G = torch.optim.Adam(Generator.parameters(), lr=lr_G)
opt_D = torch.optim.Adam(Discriminator.parameters(), lr=lr_D)

plt.ion()

for step in range(10000):
    true_data = true_result()
    G_data = Variable(torch.randn(BATCH_SIZE, N_DATA))#随机生成的
    G_data_o = Generator(G_data)#随机画出来的画
    
    prob_true = Discriminator(true_data)#将真的给D鉴定
    prob_false = Discriminator(G_data_o)#将随机画的给D鉴定
    
    G_loss = torch.mean(torch.log(1. - prob_false))#越小越接近真实
    D_loss = -torch.mean(torch.log(prob_true) + torch.log(1. - prob_false))
    
    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)
    opt_D.step()
    
    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()
    
    if 0 == step % 50:
        plt.cla()
        plt.plot(paint_points[0], G_data_o.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting')
        plt.plot(paint_points[0], 2 * np.power(paint_points[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')#t_data最大为2
        plt.plot(paint_points[0], 1 * np.power(paint_points[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')#t_data最小为1
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_true.data.numpy().mean(), fontdict={'size': 13})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((0, 3))
        plt.legend(loc='upper right', fontsize=10)
        plt.draw()
        plt.pause(0.01)

plt.ioff()
plt.show()