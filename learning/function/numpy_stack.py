# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 16:01:18 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""
import numpy as np

#垂直方向的concat
data1 = np.array([[1, 2, 3]])
print('data1:', data1.shape)
print(data1)

data2 = np.array([[2, 3, 4], [3, 4, 5]])
print('data2:', data2.shape)
print(data2)

data = np.vstack((data1, data2))#垂直 100*2, 300*2 ==> 400*2
print('data:', data.shape)
print(data)

#data1: (1, 3)
#[[1 2 3]]
#data2: (2, 3)
#[[2 3 4]
# [3 4 5]]
#data: (3, 3)
#[[1 2 3]
# [2 3 4]
# [3 4 5]]

#水平方向的concat
data1 = np.array([[1, 2, 3]])
print('data1:', data1.shape)
print(data1)

data2 = np.array([[2, 3, 4, 5, 6, 7]])
print('data2:', data2.shape)
print(data2)

data = np.hstack((data1, data2))#水平 20*10, 20*100 ==> 20*110
print('data:', data.shape)
print(data)

#data1: (1, 3)
#[[1 2 3]]
#data2: (1, 6)
#[[2 3 4 5 6 7]]
#data: (1, 9)
#[[1 2 3 2 3 4 5 6 7]]

data1 = np.array([[1, 2, 3]])
print('data1:', data1.shape)
print(data1)

data = np.stack(data1, axis=0)
print('data:', data.shape)
print(data)

data = np.stack(data1, axis=1)
print('data:', data.shape)
print(data)

#data1: (1, 3)
#[[1 2 3]]
#data: (1, 3)
#[[1 2 3]]
#data: (3, 1)
#[[1]
# [2]
# [3]]

data1 = [[1,2,3], [4,5,6]]
data = np.stack(data1, axis=0)
print('data:', data.shape)
print(data)

data = np.stack(data1, axis=1)
print('data:', data.shape)
print(data)

#data: (2, 3)
#[[1 2 3]
# [4 5 6]]
#data: (3, 2)
#[[1 4]
# [2 5]
# [3 6]]