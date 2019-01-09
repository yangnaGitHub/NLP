# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 16:08:06 2019

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

####准备数据
steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
x_np = np.sin(steps)
y_np = np.cos(steps)

plt.plot(steps, x_np, 'g-', label='sin')
plt.plot(steps, y_np, 'r-', label='cos')
plt.legend(loc='best')
plt.show()

####construct network
class secondRNN(nn.Module):
    def __init__(self):
        super(secondRNN, self).__init__()
        self.rnn = nn.RNN(
                input_size=1,
                hidden_size=32,
                num_layers=1,
                batch_first=True)
        self.out = nn.Linear(32, 1)
    
    def forward(self, x_input, h_state):
        #r_out:batch_size*time_step*hidden_size
        #h_state:num_layers*time_step*hidden_size
        r_out, h_state = self.rnn(x_input, h_state)#不是None,所以是上一次的h_state
        outs = []
        for time_step in range(r_out.size(1)):#维度1的数值==>time_step
            outs.append(self.out(r_out[:, time_step, :]))#对每time_step都进行linear之后再输出,time_step * [(batch_size*1*32)(32*1)] => batch_size*time_step*1
        #我的想法:可以这样么
        #self.out = nn.Linear(time_step*hidden_size, time_step)
        #r_out:(batch_size*time_step*hidden_size) =>reshape=> (batch_size*[time_step*hidden_size])
        #次数:(batch_size*[time_step*hidden_size])(time_step*hidden_size, time_step) => batch_size*time_step =>reshape=> batch_size*time_step*1
        return torch.stack(outs, dim=1), h_state

second = secondRNN()
print(second)

####define optimizer and loss function
optimizer = torch.optim.Adam(second.parameters(), lr=0.02)
loss_func = nn.MSELoss()

plt.figure(1, figsize=(12, 5))
plt.ion() 

TIME_STEP = 10
h_state = None
for step in range(100):
    #step_skip = 1#静态
    ####动态开始
    step_skip = np.random.randint(1, 4)#time_step不定,所以torch是动态的
    ####动态结束
    start, end = step*np.pi, (step+step_skip)*np.pi
    ministeps = np.linspace(start, end, step_skip*TIME_STEP, dtype=np.float32, endpoint=False)
    minix_np = np.sin(ministeps)
    miniy_np = np.cos(ministeps)
    #batch_szie*time_step*input_size=>(1*10*1)
    x_val, y_val = Variable(torch.from_numpy(minix_np[np.newaxis, :, np.newaxis])), Variable(torch.from_numpy(miniy_np[np.newaxis, :, np.newaxis]))
    output, h_state = second(x_val, h_state)
    h_state = h_state.data
    loss = loss_func(output, y_val)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    plt.plot(ministeps, miniy_np.flatten(), 'r-')
    plt.plot(ministeps, output.view(10*step_skip).data.numpy(), 'g-')
    plt.draw()
    plt.pause(0.05)

plt.ioff()
plt.show()