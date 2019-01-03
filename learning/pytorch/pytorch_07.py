# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 15:38:35 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

####造数据
torch.manual_seed(1)

x_data = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y_data = x_data.pow(2) + 0.1* torch.normal(torch.zeros(x_data.size()))

plt.scatter(x_data.numpy(), y_data.numpy())
plt.show()

BATCH_SIZE = 32
torch_dataset = Data.TensorDataset(x_data, y_data)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

class basic(torch.nn.Module):
    def __init__(self):
        super(basic, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.output = torch.nn.Linear(20, 1)

    def forward(self, x_input):
        return self.output(F.relu(self.hidden(x_input)))

net_sgd = basic()
net_momentum = basic()
net_rmsprop = basic()
net_adam = basic()
nets = [net_sgd, net_momentum, net_rmsprop, net_adam]

LR = 0.01
#梯度下降
opt_sgd = torch.optim.SGD(net_sgd.parameters(), lr=LR)
#梯度下降在家动量原则
opt_momentum = torch.optim.SGD(net_momentum.parameters(), lr=LR, momentum=0.8)
opt_rmsprop = torch.optim.RMSprop(net_rmsprop.parameters(), lr=LR, alpha=0.9)
opt_adam = torch.optim.Adam(net_adam.parameters(), lr=LR, betas=(0.9, 0.99))
optimizers = [opt_sgd, opt_momentum, opt_rmsprop, opt_adam]

loss_func = torch.nn.MSELoss()
losses = [[], [], [], []]

EPOCH = 12
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(loader):
        x_val,y_val = Variable(batch_x),Variable(batch_y)
        for net,opt,s_loss in zip(nets, optimizers, losses):
            output = net(x_val)
            loss = loss_func(output, y_val)
            opt.zero_grad()
            loss.backward()
            opt.step()
            s_loss.append(loss.data[0])