# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:31:43 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.Series(np.random.randn(10).cumsum(), index=np.arange(0, 100, 10))
data.plot()
#Series.plot()参数
#ax:绘图所用的matplotlib子图对象；如果没传值，则使用当前活动的matplotlib子图
#style:传给matplotlib的样式字符串，比如'ko--'
#alpha:图片不透明度(从0到1)
#kind:可以是 'area'、 'bar'、 'barh'、 'density'、'hist'、 'kde'、 'line'、 'pie'
#logy:在y轴上使用对数缩放
#use_index:使用对象索引刻度标签
#rot:刻度标签的旋转(0到360)
#xticks:用于x轴刻度的值
#yticks:用于y轴
#xlim:x轴范围(例如[0,10])
#ylim:y轴范围
#grid:展示轴网格(默认是打开的)

df = pd.DataFrame(np.random.randn(10, 4).cumsum(0), columns=['A', 'B', 'C', 'D'], index=np.arange(0, 100, 10))
df.plot()#df.plot.line( )
#DataFrame.plot()的参数
#subplots:将DataFrame的每一列绘制在独立的子图中
#sharex:如果subplots=True，则共享相同的x轴、刻度和范围
#sharey:如果subplots=True，则共享相同的y轴
#figsize:用于生成图片尺寸的元组
#title:标题字符串
#legend:添加子图图例(默认是True)
#sort_columns:按字母顺序绘制各列，默认情况下使用已有的列顺序

#plot.bar()和plot.barh()可以分别绘制垂直和水平的柱状图
fig, axes = plt.subplots(2, 1)
data = pd.Series(np.random.rand(16), index=list('abcdefghijklmnop'))
data.plot.bar(ax=axes[0], color='k', alpha=0.7)#作为X轴
data.plot.barh(ax=axes[1], color='k', alpha=0.7)#作为y轴

df = pd.DataFrame(np.random.rand(6, 4), index=['one', 'two', 'three', 'four', 'five', 'six'], columns=pd.Index(['A', 'B', 'C', 'D'], name='Genus'))
df.plot.bar()
#传递stacked=True来生成堆积柱状图
df.plot.barh(stacked=True, alpha=0.5)

import seaborn as sns
#两个不同的标准正态分布组成的双峰分布
comp1 = np.random.normal(0, 1, size=200)
comp2 = np.random.normal(10, 2, size=200)
values = pd.Series(np.concatenate([comp1, comp2]))
#distplot方法可以绘制直方图和连续密度估计
sns.distplot(values, bins=100, color='k')