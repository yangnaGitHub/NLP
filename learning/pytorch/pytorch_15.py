# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 17:19:15 2019

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import init
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

N_SAMPLES = 2000
x_data = np.linspace(-7, 10, N_SAMPLES)[:, np.newaxis]
x_noise = np.random.normal(0, 2, x_data.shape)
y_data = np.square(x_data) - 5 + x_noise

t_x_data = np.linspace(-7, 10, 200)[:, np.newaxis]
t_noise = np.random.normal(0, 2, t_x_data.shape)
t_y_data = np.square(t_x_data) - 5 + t_noise

train_x, train_y = torch.from_numpy(x_data).float(), torch.from_numpy(y_data).float()
test_x, test_y = torch.from_numpy(t_x_data).float(), torch.from_numpy(t_y_data).float()

BATCH_SIZE = 64
train_dataset = Data.TensorDataset(train_x, train_y)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
plt.scatter(train_x.numpy(), train_y.numpy(), c='red', s=50, alpha=0.2, label='train')
plt.show()

N_HIDDEN = 8
ACTIVATION = torch.tanh
B_INIT = -0.2 
class BN_net(nn.Module):
    def __init__(self, bn=False):
        super(BN_net, self).__init__()
        self.do_bn = bn#是否bn
        self.fcs = []
        self.bns = []
        self.bn_input = nn.BatchNorm1d(1, momentum=0.5)#输入的特征数=1
        
        for index in range(N_HIDDEN):
            input_size = 1 if 0 == index else 10#第一次输入是1,其他7次输入是10
            fc = nn.Linear(input_size, 10)
            setattr(self, 'fc%i' % index, fc)#pytorch一定要将层信息变成class的属性
            self._set_init(fc)
            self.fcs.append(fc)
            if self.do_bn:
                bn = nn.BatchNorm1d(10, momentum=0.5)#每一层一次bn,momentum平滑化batch的mean和stddev
                setattr(self, 'bn%i' % index, bn)
                self.bns.append(bn)
        self.predict = nn.Linear(10, 1)#输出
        self._set_init(self.predict)
    
    #初始化
    def _set_init(self, layer):
        init.normal_(layer.weight, mean=0., std=.1)#从给定均值和标准差的正态分布N(mean,std)中生成值,填充输入的张量或变量
        init.constant_(layer.bias, B_INIT)#用val的值填充输入的张量或变量
    
    def forward(self, x_input):
        pre_acivation = [x_input]
        if self.do_bn:
            x_input = self.bn_input(x_input)
        layer_input = [x_input]
        for index in range(N_HIDDEN):
            x_input = self.fcs[index](x_input)
            pre_acivation.append(x_input)
            if self.do_bn:
                x_input = self.bns[index](x_input)
            x_input = ACTIVATION(x_input)#每次做一次激励函数,bn全连接层和激励函数之间
            layer_input.append(x_input)
        #pre_acivation => 原始的输入每一层的输入
        #layer_input => pre_acivation中加上bn和激励层的输入
        return self.predict(x_input), layer_input, pre_acivation

nets = [BN_net(), BN_net(bn=True)]
LR = 0.03
opts = [torch.optim.Adam(net.parameters(), lr=LR) for net in nets]
loss_func = torch.nn.MSELoss()

def plot_histogram(l_in, l_in_bn, pre_ac, pre_ac_bn):
    for index,(ax_pa, ax_pa_bn, ax, ax_bn) in enumerate(zip(axs[0, :], axs[1, :], axs[2, :], axs[3, :])):#一共9个
        [all_a.clear() for all_a in [ax_pa, ax_pa_bn, ax, ax_bn]]#先clear
        if 0 == index:
            p_range = (-7, 10)
            the_range = (-7, 10)#原始输入
        else:
            p_range = (-4, 4)
            the_range = (-1, 1)#处理过的输入
        ax_pa.set_title('L' + str(index))
        #将多维数组降位一维,其中ravel()返回的是视图,flatten()返回一份拷贝
        ax_pa.hist(pre_ac[index].data.numpy().ravel(), bins=10, range=p_range, color='#FF9359', alpha=0.5)
        ax_pa_bn.hist(pre_ac_bn[index].data.numpy().ravel(), bins=10, range=p_range, color='#74BCFF', alpha=0.5)
        ax.hist(l_in[index].data.numpy().ravel(), bins=10, range=the_range, color='#FF9359')
        ax_bn.hist(l_in_bn[index].data.numpy().ravel(), bins=10, range=the_range, color='#74BCFF')
        for all_a in [ax_pa, ax, ax_pa_bn, ax_bn]: 
            all_a.set_yticks(())
            all_a.set_xticks(())
        ax_pa_bn.set_xticks(p_range)
        ax_bn.set_xticks(the_range)#X轴的显示
        axs[0, 0].set_ylabel('PreAct')
        axs[1, 0].set_ylabel('BN PreAct')
        axs[2, 0].set_ylabel('Act')
        axs[3, 0].set_ylabel('BN Act')
    plt.pause(0.01)
    
if __name__ == '__main__':
    fig, axs = plt.subplots(4, N_HIDDEN+1, figsize=(10, 5))#4*9个图
    plt.ion()
    plt.show()
    
    losses = [[], []]
    EPOCH = 12
    for epoch in range(EPOCH):
        layer_inputs, pre_acts = [],[]
        #每一次是li + li_bn,pa + pa_bn
        for net, loss in zip(nets, losses):
            net.eval()#eval下的bn参数会被固定
            pred, layer_input, pre_act = net(test_x)
            loss.append(loss_func(pred, test_y))
            layer_inputs.append(layer_input)
            pre_acts.append(pre_act)
            net.train()#设置成 train模式继续训练
        plot_histogram(*layer_inputs, *pre_acts)
        
        for step,(b_x, b_y) in enumerate(train_loader):
            for net,opt in zip(nets, opts):
                pred, _, _ = net(b_x)
                loss = loss_func(pred, b_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
    plt.ioff()
    plt.figure(3)
    plt.plot(losses[0], c='#FF9359', lw=3, label='Original')
    plt.plot(losses[1], c='#74BCFF', lw=3, label='Batch Normalization')
    plt.xlabel('step')
    plt.ylabel('test loss')
    plt.ylim((0, 2000))
    plt.legend(loc='bast')
    
    [net.eval() for net in nets]
    preds = [net(test_x)[0] for net in nets]
    plt.figure(4)
    plt.plot(test_x.data.numpy(), preds[0].data.numpy(), c='#FF9359', lw=4, label='Original')
    plt.plot(test_x.data.numpy(), preds[1].data.numpy(), c='#74BCFF', lw=4, label='Batch Normalization')
    plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='r', s=50, alpha=0.2, label='train')
    plt.legend(loc='best')
    plt.show()