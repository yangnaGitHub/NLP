# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 16:08:29 2019

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

#Deep Q Network
import torch
import torch.nn as nn
#from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym

env = gym.make('CartPole-v0')#什么环境
env = env.unwrapped

c_actions = env.action_space.n#可以做的动作的个数
c_states = env.observation_space.shape[0]#可以处于状态的个数

class dqn_net(nn.Module):
    def __init__(self):
        super(dqn_net, self).__init__()
        self.fc1 = nn.Linear(c_states, 10)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(10, c_actions)
        self.out.weight.data.normal_(0, 0.1)#输入是state,输出是action
        
    def forward(self, x_input):
        return self.out(F.relu(self.fc1(x_input)))

MEMORY_CAPACITY = 2000
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
BATCH_SIZE = 32
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape#看action的表达方式
class dqn(object):
    def __init__(self):
        self.eval_net, self.target_net = dqn_net(), dqn_net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, c_states*2+2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.01)
        self.loss_func = nn.MSELoss()
    
    def choose_action(self, x_input):#要做出什么动作,输入的是当前的状态
        #unsqueeze主要是对数据维度进行扩充
        x_input = torch.unsqueeze(torch.FloatTensor(x_input), 0)#N => 1*N
        if EPSILON > np.random.uniform():#90%的几率走深度学习
            actions_value = self.eval_net.forward(x_input)#将当前的状态当做输入,扔进eval网络得到每个action的
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
        else:#10%走随机
            action = np.random.randint(0, c_actions)
        action = action if 0 == ENV_A_SHAPE else action.reshape(ENV_A_SHAPE)
        return action
    
    def store_transition(self, state, action, reward, state_):
        transition = np.hstack((state, [action, reward], state_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def learn(self):
        if 0 == self.learn_step_counter % TARGET_REPLACE_ITER:
            self.target_net.load_state_dict(self.eval_net.state_dict())#100轮的时候参数同步
        self.learn_step_counter += 1
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)#选取batch个数据
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :c_states])
        b_a = torch.LongTensor(b_memory[:, c_states:c_states+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, c_states+1:c_states+2])
        b_s_ = torch.FloatTensor(b_memory[:, -c_states:])
        
        q_eval = self.eval_net(b_s).gather(1, b_a)#0,b_a[0] 1,b_a[1] ... batch,b_a[batch]
        q_next = self.target_net(b_s_).detach()#不求梯度,从当前图中分离的Variable,不进行反向传递误差
        q_target = b_r + GAMMA*q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

firstdqn = dqn()
for i_episode in range(400):#一共400个回合
    now_s = env.reset()#每次从头开始
    ep_reward = 0#每个回合的奖励
    while True:
        env.render()##渲染出当前的智能体以及环境的状态
        action = firstdqn.choose_action(now_s)#在当前状态下选择接下来的动作
        print(action)
        new_s, reward, done, info = env.step(action)#做了这个动作后,得到的下一个状态和奖励以及确认是否完成等信息
        
        #更新奖励
        x, x_dot, theta, theta_dot = new_s
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        
        firstdqn.store_transition(now_s, action, reward, new_s)
        ep_reward += reward#记录奖励
        if MEMORY_CAPACITY < firstdqn.memory_counter:
            firstdqn.learn()
            if done:
                print('Ep: ', i_episode, '| Ep_r: ', round(ep_reward, 2))
        if done:#如果完成后就退出这回合
            break
        now_s = new_s#更新状态
        
#Spark机器学习进阶实战
#Spark机器学习：核心技术与实践

####gather的用法
#b = torch.Tensor([[1,2,3],[4,5,6]])
#print(b)
#index_1 = torch.LongTensor([[0,1],[2,0]])
#index_2 = torch.LongTensor([[0,1,1],[0,0,0]])
##0_,0 0_,1 => 1 2
##1_,2 1_,0    6 4
#print(torch.gather(b, dim=1, index=index_1))
##0,0_ 1,1_ 1,2_ => 1 5 6
##0,0_ 0,1_ 0,2_    1 2 3
#print(torch.gather(b, dim=0, index=index_2))