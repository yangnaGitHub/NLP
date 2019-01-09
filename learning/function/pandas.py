# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 13:56:03 2019

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import pandas as pd
import numpy as np

sample = pd.DataFrame({'id':[1,1,1,3,4,5],
                       'name':['Bob','Bob','Mark','Miki','Sully','Rose'],
                       'score':[99,99,87,77,77,np.nan],
                       'group':[1,1,1,2,1,2],})
####重复值处理=>直接删除是重复数据处理的主要方法
sample[sample.duplicated()]#查看重复数据,通过drop_duplicates去掉重复
#其中drop_duplicates还可以根据某列去重,比如sample.drop_duplicates('id')

####缺失值处理=>填补
# 当缺失值少于20%时,连续变量可以使用均值或中位数填补,分类变量不需要填补,单算一类即
# 当缺失值处于20%-80%时,填补方法同上,每个有缺失值的变量可以生成一个指示哑变量,参与后续的建模
# 当缺失值多于80%时,每个有缺失值的变量可以生成一个指示哑变量,参与后续的建模,不使用原始变量
#1>查看缺失情况
sample.apply(lambda col:sum(col.isnull())/col.size)#每列缺失的比率
#2>以指定值填补
sample.score.fillna(sample.score.mean())#fillna方法完成对缺失值的填补,填补方法为均值
sample.score.fillna(sample.score.median())#以分位数
#3>缺失值指示变量
sample.score.isnull()
sample.score.isnull().apply(int)#装换成int类型

####噪声值处理=>单变量:盖帽法,分箱法,多变量:聚类法
#1>盖帽法将某连续变量均值上下三倍标准差范围外的记录替换为均值上下三倍标准差值,就是u+/-3*sigma外面的截掉
def cap(x_col, quantile=[0.01, 0.99]):
    '''盖帽法处理异常值
    Args:
        x_col：pd.Series列，连续变量
        quantile：指定盖帽法的上下分位数范围
    '''
    #生成分位数
    Q01,Q99=x_col.quantile(quantile).values.tolist()
    #替换
    if Q01 > x_col.min():
        x_col = x_col.copy()
        x_col.loc[x_col<Q01] = Q01
    
    if Q99 < x_col.min():
        x_col = x_col.copy()
        x_col.loc[x_col>Q99] = Q01
    
    return (x_col)
sample = pd.DataFrame({'normal':np.random.randn(1000)})
sample.hist(bins=50)#直方图
new = sample.apply(cap, quantile=[0.01,0.99])
new.hist(bins=50)
#2>考察数据的近邻来光滑有序数据的值,有序值分布到一些桶或箱中,qcut函数提供了分箱的实现方法
# 4 8 15 21 21 24 25 28 34
# 等深分箱,每个分箱中的样本量一致
#  1:[4 8 15]  2:[21 21 24]  3:[25 28 34]
sample =pd.DataFrame({'normal':np.random.randn(10)})
pd.cut(sample.normal, bins=sample.normal.quantile([0,0.5,1]), include_lowest=True)
# 等宽分箱,每个分箱中的取值范围一致,直方图其实首先对数据进行了等宽分箱,再计算频数画图
#  1:[4 8]  2:[15 21 21 24]  3:[25 28 34]
pd.cut(sample.normal, 5)
pd.cut(sample.normal, bins=5, labels=[1,2,3,4,5])#可以使用labels参数指定分箱后各个水平的标签
#3>聚类法=>快速聚类法将数据对象分组成为多个簇