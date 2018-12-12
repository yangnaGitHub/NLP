# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 11:36:02 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

#安装https://pytorch.org/get-started/locally/
#windows 无GPU 3.6
# conda install pytorch-cpu torchvision-cpu -c pytorch

# pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-win_amd64.whl
# pip3 install torchvision

import torch
import numpy as np

#####Torch和numpy的比较
#相互转换
data = np.arange(6).reshape((2, 3))
t_data = torch.from_numpy(data)
tensor2array = t_data.numpy()
print(data)
print(t_data)
print(tensor2array)

#torch和numpy的对比
#torch的对象只能是tensor
data = [-1, -2, 1, 2]
t_data = torch.FloatTensor(data)
print(np.abs(data))
print(torch.abs(t_data))
print(np.sin(data))
print(torch.sin(t_data))
print(np.mean(data))
print(torch.mean(t_data))
data = [[1, 2], [3, 4]]#list
t_data = torch.FloatTensor(data)
print(np.matmul(data, data))
print(torch.matmul(t_data, t_data))

#numpy可以的,torch不行的
data = np.array(data)#转换成np.array ==> 这种类型有自己的方法
print(data.dot(data))#调用自己的dot方法
print(t_data.dot(t_data))#报错

#####Variable
#variable是一个空间,你可以往里面放东西
#如果用一个variable运算的话返回的也是一个同类型的variable
from torch.autograd import Variable#Variable在这个包中
#数据
t_data = torch.FloatTensor([[1, 2], [3, 4]])
#放置数据的地方,requires_grad是参不参与误差反向传播,要不要计算梯度
t_location = Variable(t_data, requires_grad=True)
print(t_data)
print(t_location)

#比较tensor的计算和variable的计算
#variable计算是在搭建一个庞大的系统,叫做计算图
#计算图是将所有的计算步骤都连接起来,最后进行误差反向传递的时候,一次性将所有variable里面的修改梯度都计算出来(tensor没有这个能力)
t_out = torch.mean(t_data*t_data)
v_out = torch.mean(t_location*t_location)
print(t_out)
print(v_out)
v_out.backward()#模拟误差反向传递
print(t_location.grad)#t_location的梯度

#直接print只会输出variable形式的数据,很多时候要转化成tensor形式
print(t_location.data)
print(t_location.data.numpy())#numpy形式,可以用于画图等

#####激励函数
import torch.nn.functional as F
x_data = torch.linspace(-5, 5, 200)#tensor
x_val = Variable(x_data)#Variable
x_np = x_val.data.numpy()#numpy
#几种常用的激励函数
#relu
y_relu = F.relu(x_val).data.numpy()#这儿使用的是Variable,可否使用tensor呢
y_t_relu = Variable(F.relu(x_data)).data.numpy()
#sigmoid
y_sigmoid = F.sigmoid(x_val).data.numpy()
#tanh
y_tanh = F.tanh(x_val).data.numpy()
y_softplus = F.softplus(x_val).data.numpy()
#y_softmax = F.softmax(x_val).data.numpy()
import matplotlib.pyplot as plt
plt.figure(1, figsize=(10, 6))
plt.subplot(511)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(512)
plt.plot(x_np, y_t_relu, c='red', label='t_relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(513)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(514)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(515)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()