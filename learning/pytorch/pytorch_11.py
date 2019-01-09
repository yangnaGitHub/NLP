# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 16:08:22 2019

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

#大的数据可以压缩传输下载之后再解压还原
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np

####准备数据
train_data = torchvision.datasets.MNIST(
        root='./mnist',#保存的地方
        train=True,
        transform=torchvision.transforms.ToTensor(),#转换成torch.FloatTensor(C,H,W),训练的时候 normalize到[0.0, 1.0]区间
        download=False#没下载就下载,下载好了就是False
        )
BATCH_SIZE = 64
train_loader = Data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
#AutoEncoder=>[encoder, decoder]=压缩后得到压缩的特征值,再从压缩的特征值解压成原图片
class AutoEncode(nn.Module):
    def __init__(self):
        super(AutoEncode, self).__init__()
        self.encoder = nn.Sequential(
                nn.Linear(28*28, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 12),
                nn.Tanh(),
                nn.Linear(12, 3))
        self.decoder = nn.Sequential(
                nn.Linear(3, 12),
                nn.Tanh(),
                nn.Linear(12, 64),
                nn.Tanh(),
                nn.Linear(64, 128),
                nn.Tanh(),
                nn.Linear(128, 28*28),
                nn.Sigmoid())
    def forward(self, x_input):
        encode = self.encoder(x_input)
        decode = self.decoder(encode)
        return encode, decode

autoencoder = AutoEncode()

####定义优化函数和LOSS
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)
loss_func = nn.MSELoss()

N_TEST_IMG = 5#只展示前面的5个
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))#2行5列
view_data = train_data.train_data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.
#第一行
for index in range(N_TEST_IMG):
    a[0][index].imshow(np.reshape(view_data.data.numpy()[index], (28, 28)), cmap='gray')
    a[0][index].set_xticks(())
    a[0][index].set_yticks(())

for epoch in range(10):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_x_d = Variable(batch_x.view(-1, 28*28)),Variable(batch_x.view(-1, 28*28))
        #batch_y = Variable(batch_y)
        
        encode, decode = autoencoder(batch_x)
        loss = loss_func(decode, batch_x_d)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if 0 == step % 100:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
            _, decoded_data = autoencoder(view_data)
            #第二行
            for index in range(N_TEST_IMG):
                a[1][index].clear()
                a[1][index].imshow(np.reshape(decoded_data.data.numpy()[index], (28, 28)), cmap='gray')
                a[1][index].set_xticks(())
                a[1][index].set_yticks(())
            plt.draw()
            plt.pause(0.05)

plt.ioff()
plt.show()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
view_data = train_data.train_data[:200].view(-1, 28*28).type(torch.FloatTensor)/255.
encoded_data, _ = autoencoder(view_data)
fig = plt.figure(2)
ax = Axes3D(fig)
DX, DY, DZ = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
values = train_data.train_labels[:200].numpy()
for dx, dy, dz, dv in zip(DX, DY, DX, values):
    cb = cm.rainbow(int(255*dv/9))#颜色
    ax.text(dx, dy, dz, dv, backgroundcolor=cb)
ax.set_xlim(DX.min(), DX.max())
ax.set_ylim(DY.min(), DY.max())
ax.set_zlim(DZ.min(), DZ.max())
plt.show()
        