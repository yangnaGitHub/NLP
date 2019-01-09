# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 15:38:37 2018

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

#http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
#http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
#http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
#http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
#训练数据
train_data = torchvision.datasets.MNIST(
        root='./mnist',#保存的地方
        train=True,
        transform=torchvision.transforms.ToTensor(),#转换成torch.FloatTensor(C,H,W),训练的时候 normalize到[0.0, 1.0]区间
        download=True#没下载就下载,下载好了就是False
        )

#测试数据
test_data = torchvision.datasets.MNIST(root='./mnist', train=False)
BATCH_SIZE = 50
train_loader = Data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
#节省时间所以只测试前面的2000个,可以按照train的一样全部测试
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:2000]/255.
#!!!将数据的形式变成GPU能读的形式
#test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:2000].cuda()/255.
#test_y = test_data.test_labels[:2000].cuda()
test_y = test_data.test_labels[:2000].data.numpy()

####构建网络
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
                #16 * [(28+2*2-5)/1 + 1] * [(28+2*2-5)/1 + 1] = 16*28*28
                nn.Conv2d(
                    in_channels=1,#(1*28*28)
                    out_channels=16,#(16*28*28)
                    kernel_size=5,#filter size
                    stride=1,#step
                    padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2))#16*14*14
        self.conv2 = nn.Sequential(
                nn.Conv2d(16, 32, 5, 1, 2),#16*14*14 ==> 32*14*14
                nn.ReLU(),
                nn.MaxPool2d(2))#32*7*7
        self.output = nn.Linear(32*7*7, 10)#全连接层
    
    def forward(self, x_input):
        x_data = self.conv2(self.conv1(x_input))
        x_data = x_data.view(x_data.size(0), -1)#展开成batch_size * 32*7*7的数据
        return self.output(x_data), x_data

firstCNN = CNN()
#!!!GPU加速
#firstCNN.cuda()
print(firstCNN)

####可视化
from matplotlib import cm
try: 
    from sklearn.manifold import TSNE
    #manifold流形学习方法,从高维采样数据中恢复低维流形结构
    #一遍聚类一遍降维,比较好的自动聚类方法
    HAS_SK = True
except:
    HAS_SK = False
    print('Please install sklearn for layer visualization')

def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    plt.show()
    plt.pause(0.01)
plt.ion()

####定义优化和损失
optimizer = torch.optim.Adam(firstCNN.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

EPOCH = 1
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        x_val,y_val = Variable(batch_x),Variable(batch_y)
        #!!!GPU加速
        #x_val,y_val = x_val.cuda(),y_val.cuda()
        output = firstCNN(x_val)[0]
        loss = loss_func(output, y_val)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if 0 == step % 50:
            test_o, last = firstCNN(test_x)
            pred_y = torch.max(test_o, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            #!!!GPU加速
            #pred_y = torch.max(test_o, 1)[1].cuda().data.squeeze()
            #accuracy = torch.sum(pred_y == test_y) / test_y.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            if HAS_SK:
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)#n_components降到2维
                plot_only = 500
                low_dim_embs = tsne.fit_transform(last.data.numpy()[:plot_only, :])
                labels = test_y[:plot_only]
                plot_with_labels(low_dim_embs, labels)
plt.ioff()

test_output = firstCNN(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
#!!!GPU加速
#pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
