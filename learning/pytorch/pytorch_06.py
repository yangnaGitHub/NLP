# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 15:38:26 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import torch
#DataLoader用来包装数据的工具
import torch.utils.data as Data

torch.manual_seed(1)

x_data = torch.linspace(1, 10, 10)#1-10共10个
y_data = torch.linspace(10, 1, 10)#10-1共10个

BATCH_SIZE = 5
#现将数据转换成torch识别的数据集
torch_dataset = Data.TensorDataset(x_data, y_data)
#将数据集放到DataLoader中
loader = Data.DataLoader(
        dataset = torch_dataset,
        batch_size=BATCH_SIZE,#mini 向上取整(batch size,x_data.size() / BATCH_SIZE),最后一次不够,返回剩下的数据就好了
        shuffle=True,#打乱
        num_workers=2)#多线程

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ', batch_x.numpy(), '| batch y: ', batch_y.numpy())