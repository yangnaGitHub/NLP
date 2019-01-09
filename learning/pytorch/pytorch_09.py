# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 19:26:35 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

####准备数据
torch.manual_seed(1)

#训练数据
train_data = torchvision.datasets.MNIST(
        root='./mnist',#保存的地方
        train=True,
        transform=torchvision.transforms.ToTensor(),#转换成torch.FloatTensor(C,H,W),训练的时候 normalize到[0.0, 1.0]区间
        download=False#没下载就下载,下载好了就是False
        )

#测试数据
test_data = torchvision.datasets.MNIST(root='./mnist', train=False)
BATCH_SIZE = 64
train_loader = Data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels[:2000].data.numpy()

#(input0, state0) => LSTM => (output0, state1)
#(input1, state1) => LSTM => (output1, state2)
# ...
#(inputN, stateN) => LSTM => (outputN, stateN+1)
class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        #batch_size*time_step*hidden_size ==> (batch*28*64)
        self.rnn = nn.LSTM(
                input_size=28,
                hidden_size=64,#cell的个数
                num_layers=1,#几层
                batch_first=True)#nn.RNN()
        self.output = torch.nn.Linear(64, 10)#(batch*28*10)
    
    def forward(self, x_input):
        #x_input:batch_size*time_step*input_size
        #所有time_step
        #r_out:batch_size*time_step*hidden_size(10)
        #最后一个time_step
        #h_n:(num_layers*batch_size*hidden_size)
        #h_c:(num_layers*batch_size*hidden_size)
        r_out, (h_n, h_c) = self.rnn(x_input, None)#None表示hidden state会用全0的state
        return self.output(r_out[:, -1, :])#最后一个time_step的h_n

firstRNN = RNN()
print(firstRNN)

####定义优化和损失
optimizer = torch.optim.Adam(firstRNN.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()

####train
for epoch in range(1):
    for step,(batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.view(-1, 28, 28)
        batch_x, batch_y = Variable(batch_x),Variable(batch_y)
        output = firstRNN(batch_x)
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if 0 == step % 50:
            test_output = firstRNN(test_x.view(-1, 28, 28))
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

test_output = firstRNN(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
