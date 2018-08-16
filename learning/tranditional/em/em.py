# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 16:56:42 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import numpy as np
import copy
import math

def construdt_data(count=100):
    return [np.random.randn() for index in range(count)]

def init_params(kinds=2, sigmalow=0, sigmahigh=50):
    mu = [np.random.randn() for index in range(kinds)]
    sigma = [np.random.randint(sigmalow, sigmahigh) for index in range(kinds)]
    return mu, sigma

def E_step(data, mu, sigma, expect):
    for index, value in enumerate(data):
        total = 0
        for othindex in range(len(mu)):
            total += math.exp((-1 / (2 * (float(sigma[othindex] ** 2)))) * (float(value - mu[othindex])) ** 2)
        for othindex in range(len(mu)):
            signal = math.exp((-1 / (2 * (float(sigma[othindex] ** 2)))) * (float(value - mu[othindex])) ** 2)
            expect[index, othindex] = signal / total

def M_step(mu, sigma, expect):
    for othindex in range(len(mu)):
        signal = 0
        total = 0
        sigma_calc = 0.0
        for index, value in enumerate(data):
            signal += expect[index, othindex] * value
            total += expect[index, othindex]
            mu[othindex] = signal / total
            sigma_calc += expect[index, othindex] * ((value - mu[othindex]) ** 2)
            sigma[othindex] = math.sqrt(sigma_calc/total)

if __name__ == '__main__':
    data = construdt_data(count=200)
    total_count = len(data)
    mu, sigma = init_params(kinds=2)
    print(mu)
    print(sigma)
    index = 0
    Epsilon = 0.0001
    expect = np.zeros((total_count, len(mu)))
    while True:
        Old_mu = copy.deepcopy(mu)
        E_step(data, mu, sigma, expect)
        M_step(mu, sigma, expect)
        
        if sum(abs(np.array(mu) - np.array(Old_mu))) < Epsilon:
            break
        
        index += 1
        if index >= 10000:
            break
    print(mu)
    print(sigma)

#import math
#import copy
#import numpy as np
#import matplotlib.pyplot as plt
#
#def ini_data(Sigma, Mu1, Mu2, k, N):#6, 40, 20, 2, 15
#    global X
#    global Mu
#    global Expectations
#    X = np.zeros((1, N))#1*15 ==> 数据X
#    Mu = np.random.random(2)# ==> 均值
#    Expectations = np.zeros((N, k))#15 * 2
#    inputdata = [-67,-48,6,8,14,16,23,24,28,29,41,49,56,60,75]#15
#    for i in range(0,len(inputdata)):
#        X[0,i] = inputdata[i]
##X:[-67,-48,6,8,14,16,23,24,28,29,41,49,56,60,75]
##Mu + Expectations:随机
#
#def e_step(Sigma, k, N):#6 * 2 *15
#    global Expectations
#    global Mu
#    global X
#    for i in range(0, N):
#        Denom = 0
#        for j in range(0, k):#参数Sigma, Mu已知
#            Denom += math.exp((-1 / (2 * (float(Sigma ** 2)))) * (float(X[0, i] - Mu[j])) ** 2)
#        for j in range(0, k):
#            Numer = math.exp((-1 / (2 * (float(Sigma ** 2)))) * (float(X[0, i] - Mu[j])) ** 2)
#            Expectations[i, j] = Numer / Denom
#
#def m_step(k, N):
#    global Expectations
#    global X
#    for j in range(0, k):
#        Numer = 0
#        Denom = 0
#        for i in range(0, N):
#            Numer += Expectations[i, j] * X[0, i]
#            Denom += Expectations[i, j]
#        Mu[j] = Numer / Denom
#        # 算法迭代iter_num次，或达到精度Epsilon停止迭代
#
#
#def run(Sigma, Mu1, Mu2, k, N, iter_num, Epsilon):
#    #初始化数据
#    #方差Sigma, Mu1, Mu2
#    ini_data(Sigma, Mu1, Mu2, k, N)#6, 40, 20, 2, 15
#    print('初始<u1,u2>:', Mu)
#    
#    #train
#    for i in range(iter_num):#迭代1000轮
#        Old_Mu = copy.deepcopy(Mu)
#        #E step
#        e_step(Sigma, k, N)#6 * 2 * 15
#        #M step
#        m_step(k, N)
#        print(i, Mu)
#        if sum(abs(Mu - Old_Mu)) < Epsilon:#0.0001
#            break
#
#if __name__ == '__main__':
#    run(6, 40, 20, 2, 15, 1000, 0.0001)
#    plt.hist(X[0, :], 50)
#    plt.show()
    

#from __future__ import division
#from numpy import *
#import math as mt
##首先生成一些用于测试的样本
##指定两个高斯分布的参数，这两个高斯分布的方差相同
#sigma = 6
#miu_1 = 40
#miu_2 = 20
# 
##随机均匀选择两个高斯分布，用于生成样本值
#N = 1000
#X = zeros((1, N))
#for i in range(N):
#    if random.random() > 0.5:#使用的是numpy模块中的random
#        X[0, i] = random.randn() * sigma + miu_1
#    else:
#        X[0, i] = random.randn() * sigma + miu_2
# 
##上述步骤已经生成样本
##对生成的样本，使用EM算法计算其均值miu
# 
##取miu的初始值
#k = 2
#miu = random.random((1, k))
##miu = mat([40.0, 20.0])
#Expectations = zeros((N, k))
# 
#for step in range(1000):#设置迭代次数
#    #步骤1，计算期望
#    for i in range(N):
#        #计算分母
#        denominator = 0
#        for j in range(k):
#            denominator = denominator + mt.exp(-1 / (2 * sigma ** 2) * (X[0, i] - miu[0, j]) ** 2)
#        
#        #计算分子
#        for j in range(k):
#            numerator = mt.exp(-1 / (2 * sigma ** 2) * (X[0, i] - miu[0, j]) ** 2)
#            Expectations[i, j] = numerator / denominator
#    
#    #步骤2，求期望的最大
#    #oldMiu = miu
#    oldMiu = zeros((1, k))
#    for j in range(k):
#        oldMiu[0, j] = miu[0, j]
#        numerator = 0
#        denominator = 0
#        for i in range(N):
#            numerator = numerator + Expectations[i, j] * X[0, i]
#            denominator = denominator + Expectations[i, j]
#        miu[0, j] = numerator / denominator
#        
#    
#    #判断是否满足要求
#    epsilon = 0.0001
#    if sum(abs(miu - oldMiu)) < epsilon:
#        break
#    
#    print(step)
#    print(miu)
#    
#print(miu)

#from ..utils.tools import Solver
#
#import numpy as np
#import copy
#
#class EM(Solver):
#    """
#    this algorithm just require to lean the Gauss distribution elements 'mu' and 'sigma'
#    """
#    def __init__(self, max_iter=100, theta=1e-5):
#        self.max_iter = max_iter
#        self.theta = theta
#
#    def _init_parameters(self, X):
#        rows, cols = X.shape
#        mu_init = np.nanmean(X, axis=0)#含有nan的求和
#        sigma_init = np.zeros((cols, cols))
#        for i in range(cols):
#            for j in range(i, cols):
#                vec_col = X[:, [i, j]]
#                vec_col = vec_col[~np.any(np.isnan(vec_col), axis=1), :].T
#                if len(vec_col) > 0:
#                    cov = np.cov(vec_col)
#                    cov = cov[0, 1]
#                    sigma_init[i, j] = cov
#                    sigma_init[j, i] = cov
#
#                else:
#                    sigma_init[i, j] = 1.0
#                    sigma_init[j, i] = 1.0
#
#        return mu_init, sigma_init
#
#    def _e_step(self, mu,sigma, X):
#        samples,_ = X.shape
#        for sample in range(samples):
#            if np.any(np.isnan(X[sample,:])):
#                loc_nan = np.isnan(X[sample,:])
#                new_mu = np.dot(sigma[loc_nan, :][:, ~loc_nan],
#                                np.dot(np.linalg.inv(sigma[~loc_nan, :][:, ~loc_nan]),
#                                       (X[sample, ~loc_nan] - mu[~loc_nan])[:,np.newaxis]))
#                nan_count = np.sum(loc_nan)
#                X[sample, loc_nan] = mu[loc_nan] + new_mu.reshape(1,nan_count)
#
#        return X
#
#    def _m_step(self,X):
#        rows, cols = X.shape
#        mu = np.mean(X, axis=0)
#        sigma = np.cov(X.T)
#        tmp_theta = -0.5 * rows * (cols * np.log(2 * np.pi) +
#                                  np.log(np.linalg.det(sigma)))
#
#        return mu, sigma,tmp_theta
#
#    def solve(self, X):
#        mu, sigma = self._init_parameters(X)
#        complete_X,updated_X = None, None
#        rows,_ = X.shape
#        theta = -np.inf
#        for iter in range(self.max_iter):
#            updated_X = self._e_step(mu=mu, sigma=sigma, X=copy.copy(X))
#            mu, sigma, tmp_theta = self._m_step(updated_X)
#            for i in range(rows):
#                tmp_theta -= 0.5 * np.dot((updated_X[i, :] - mu),
#                                          np.dot(np.linalg.inv(sigma), (updated_X[i, :] - mu)[:, np.newaxis]))
#            if abs(tmp_theta-theta)<self.theta:
#                complete_X = updated_X
#                break;
#            else:
#                theta = tmp_theta
#        else:
#            complete_X = updated_X
#        return complete_X