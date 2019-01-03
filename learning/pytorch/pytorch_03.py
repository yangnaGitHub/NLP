# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 21:15:48 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

####1.准备数据
x_data = torch.ones(100, 2)#x_data.size() ==> torch.Size([100, 2])
x_data_0 = torch.normal(2*x_data, 1)
y_data_0 = torch.zeros(100)
x_data_1 = torch.normal(-2*x_data, 1)
y_data_1 = torch.ones(100)

#100*2 + 100*2 ==> 200*2
x_cat = torch.cat((x_data_0, x_data_1), 0).type(torch.FloatTensor)#torch.FloatTensor 32bit
y_cat = torch.cat((y_data_0, y_data_1), ).type(torch.LongTensor)#torch.LongTensor 64bit

#torch只能在Variable上训练
x_v = Variable(x_cat)
y_v = Variable(y_cat)

# 画图
plt.scatter(x_v.data.numpy()[:, 0], x_v.data.numpy()[:, 1], c=y_v.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()

####2.construct network
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)
    
    def forward(self, x_input):
        return self.output(F.relu(self.hidden(x_input)))

second = Net(n_feature=2, n_hidden=10, n_output=2)
print(second)

####3.define train process
optimizer = torch.optim.SGD(second.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()
plt.show()
 
for times in range(100):
    output = second(x_v)#200*2
    loss = loss_func(output, y_v)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
####4.display
    if 0 == times % 5:
        plt.cla()
        predict = torch.max(F.softmax(output), 1)[1]
        pred_y = predict.data.numpy().squeeze()
        target_y = y_v.data.numpy()
        plt.scatter(x_v.data.numpy()[:, 0], x_v.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 15, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()