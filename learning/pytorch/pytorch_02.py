# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 18:58:58 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

#torch.Size([100]) ==> torch.Size([100, 1])
x_data = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y_data = x_data.pow(2) + 0.2*torch.rand(x_data.size())

v_x, v_y = Variable(x_data), Variable(y_data)#其实tensor也可以data.numpy(),为何偏要使用Variable来做??
plt.scatter(x_data.data.numpy(), y_data.data.numpy())
plt.show()

#第一步:构建NET
import torch.nn.functional as F
class firstNET(torch.nn.Module):#继承
    #这个函数一般是初始化所有的层对象,有积木
    def __init__(self, n_feature, n_hidden, n_output):
        super(firstNET, self).__init__()
        #torch.nn.Linear这是一个类=>是一个线性操作申请w(n_feature, n_hidden) + b(n_hidden)
        self.hidden = torch.nn.Linear(n_feature, n_hidden)#这是一个对象
        self.output = torch.nn.Linear(n_hidden, n_output)
    
    #这个是构建网络,搭建积木
    def forward(self, x_input):#继承的forward的方法
        x_r = F.relu(self.hidden(x_input))#为啥对象是直接传参数,是默认调用的forward么??
        #x_f_r = F.relu(self.hidden.forward(x_data))
        x_o = self.output(x_r)
        #x_f_o = self.output(x_f_r)
        #print(x_o, x_f_o)#验证是不是默认调用forward方法
        return x_o
#网络结构:(100*1)*(1*10)+b=>(100*10)*(10*1)=>(100*1)
first = firstNET(n_feature=1, n_hidden=10, n_output=1)
print(first)

#matplotlib有两种显示模式 =>阻塞 + 交互
#命令行默认是交互的,脚本中默认是阻塞的
plt.ion()#打开交互模式ioff()关掉交互模式
plt.show()

#第二步:确定怎么计算loss和怎么优化
print(first.parameters())
optimizer = torch.optim.SGD(first.parameters(), lr=0.5)
f_loss = torch.nn.MSELoss()#使用什么损失函数,这也是一个对象

#第三步:训练
for t_s in range(100):
    #用Variable的和不用Variable的没有啥区别,感觉
#    r_first = first(v_x)#直接调用的是forward方法=>这是扔进去数据进行训练,这是网络的输出(x_o)
#    r_loss = f_loss(r_first, v_y)#也是调用forward方法,预测的和真实的MSE
    r_first = first(x_data)
    r_loss = f_loss(r_first, y_data)
    optimizer.zero_grad()#清空上一步的残余更新参数值
    r_loss.backward()#误差反向传播,计算参数更新值 <== 先反向传播一下
    optimizer.step()#参数更新 <== 在更新参数
    
    if 0 == (t_s % 5):
        plt.cla()#清除轴
#        plt.clf()#清除整个当前图片
#        plt.close()#关上画图窗户
        plt.scatter(v_x.data.numpy(), v_y.data.numpy())
        plt.plot(v_x.data.numpy(), r_first.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % r_loss.data, fontdict={'size': 13, 'color':'red'})
        plt.pause(0.1)#暂停