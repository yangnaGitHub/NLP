# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 15:34:43 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import tensorflow as tf
import numpy as np
import gym
import random
from conf.conf import args
from collections import deque
#import os

def get_section_name():
    string = __file__
    startpos = string.rfind('/')
    endpos = string.rfind('.py')
    global section
    section = string[startpos+1:endpos]

def get_conf(name, wclass='str'):
    return args.get_option(section, name, wclass)

class DQN():
    def __init__(self, env):
        self.epsilon_step = (get_conf('init_epsilon', 'float') - get_conf('final_epsilon', 'float')) / 10000
        self.action_dim = env.action_space.n
        self.state_dim = env.observation_space.shape[0]
        print(env.observation_space, self.action_dim, self.state_dim)
        
        self.neuron_num = 100
        self.replay_buffer = deque()
        self.epsilon = get_conf('init_epsilon', 'float')
        self.sess = tf.InteractiveSession()
        self.init_network()
        self.sess.run(tf.global_variables_initializer())
    #x*state_dim => (hidden_layer)x*self.neuron_num => (Q_value)x*self.action_dim =>
    def init_network(self):
        self.input_layer = tf.placeholder(tf.float32, [None, self.state_dim])#x*state_dim
        self.action_input = tf.placeholder(tf.float32, [None, self.action_dim])
        self.y_input = tf.placeholder(tf.float32, [None])#x
        
        W1 = tf.Variable(tf.truncated_normal([self.state_dim, self.neuron_num], stddev=0.1), name='W1')
        b1 = tf.Variable(tf.constant(0.1, shape=[self.neuron_num]), name='b1')
        hidden_layer = tf.nn.relu(tf.matmul(self.input_layer, W1 ) + b1)
        
        W2 = tf.Variable(tf.truncated_normal([self.neuron_num, self.action_dim], stddev=0.1), name='W2')
        b2 = tf.Variable(tf.constant(0.1, shape=[self.action_dim]), name='b2')
        self.Q_value = tf.matmul(hidden_layer, W2) + b2
        
        value = tf.reduce_sum(self.Q_value*self.action_input, reduction_indices=1)
        print('value:{}'.format(value.shape))
        self.cost = tf.reduce_mean(tf.square(value - self.y_input))
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(self.cost)
    
    def percieve(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros([self.action_dim])
        one_hot_action[action] = 1

        self.replay_buffer.append([state, one_hot_action, reward, next_state, done])

        if len(self.replay_buffer) > get_conf('replay_size', 'int'):
            self.replay_buffer.popleft()
        if len(self.replay_buffer) > get_conf('train_state_size', 'int'):
            self.train()

    def train(self):
        batch_size = get_conf('batch_size', 'int')
        mini_batch = random.sample(self.replay_buffer, batch_size)
        state_batch = [data[0] for data in mini_batch]
        action_batch = [data[1] for data in mini_batch]
        reward_batch = [data[2] for data in mini_batch]
        next_state_batch = [data[3] for data in mini_batch]
        done_batch =  [data[4] for data in mini_batch]

        y_batch = []

        next_state_reward = self.Q_value.eval(feed_dict={self.input_layer:next_state_batch})

        gamma = get_conf('gamma', 'float')
        for i in range(batch_size):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + gamma*np.max(next_state_reward[i]))


        self.optimizer.run(
            feed_dict = {
                self.input_layer:state_batch,
                self.action_input:action_batch,
                self.y_input:y_batch
            }
        )
                            
    def get_greedy_action(self, state):
        value = self.Q_value.eval(feed_dict={self.input_layer:[state]})[0]
        return np.argmax(value)

    def get_action(self, state):
        if self.epsilon > get_conf('final_epsilon', 'float'):
            self.epsilon -= self.epsilon_step
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        else:
            return self.get_greedy_action(state)

if __name__ == '__main__':
    #args.listAlloptions()
    #有监督学习的评测工具是数据,强化学习的评测工具是环境
    #需要提供一个环境给Agent运行,才能评测Agent的策略的优劣
    get_section_name()
    
    env = gym.make(get_conf('env_name'))
    
    #动作空间
    print(env.action_space)
    #状态空间
    print(env.observation_space)
    #状态空间的每一维度的最值
    print(env.observation_space.high)
    print(env.observation_space.low)
    #
    
    agent = DQN(env)
    
    for episode in range(get_conf('episode', 'int')):#10000
        total_reward = 0
        state = env.reset()#重置环境,环境的initial observation重置
        for step in range(get_conf('step', 'int')):#10000
            env.render()#渲染出当前的智能体以及环境的状态
            action = agent.get_action(state)
            #env.step()会返回四个值:observation(object),reward(float),done(boolean),info(dict)
            #done表示是否应该reset环境
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            agent.percieve(state, action, reward, next_state, done)
            if done:
                break
            state = next_state
        print('total reward this episode is: {}'.format(total_reward))