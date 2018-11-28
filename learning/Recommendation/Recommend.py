# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:59:48 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

#数据集下载地址=>https://grouplens.org/datasets/movielens/

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

base_path = './movielens_1m/'
df = pd.read_csv(base_path + 'ratings.csv', sep=',', skiprows=1, names=['user_id', 'item_id', 'rating', 'timestamp'])

movies = pd.read_csv(base_path + 'movies.csv', sep=',', skiprows=1, names=['item_id', 'title', 'genres'])

df = pd.merge(df, movies, on='item_id')
#user_id(用户id)  item_id(电影id)  rating(评分)   timestamp(评分时间)             title(电影名称)         genres(电影类别)
#数据集的简要描述
df.describe()#平均数,中位数等
df.info()

rating = pd.DataFrame(df.groupby('title')['rating'].mean())
rating['number_of_ratings'] = df.groupby('title')['rating'].count()

import matplotlib.pyplot as plt
rating['rating'].hist(bins=50)#分值的直方图
rating['number_of_ratings'].hist(bins=60)

#探索电影评分和被评分次数之间的关系
import seaborn as sns
#设置阈值
sns.jointplot(x='rating', y='number_of_ratings', data=rating)#平均评分和被评分次数之间呈正相关关系
#基于项目的推荐系统
movies_matrix = df.pivot_table(index='user_id', columns='title', values='rating')

rating.sort_values('number_of_ratings', ascending=False).head(10)

FG_user_rating = movies_matrix['Forrest Gump (1994)']#user_id(每个用户给这部电影的打分)
SR_user_rating = movies_matrix['Shawshank Redemption, The (1994)']
FG_user_rating.head()
SR_user_rating.head()

#corrwith功能计算两个dataframe对象的行或列的两两相关关系
similar_to_FG = movies_matrix.corrwith(FG_user_rating)#电影与其他电影的相关性
similar_to_FG.head()
similar_to_SR = movies_matrix.corrwith(SR_user_rating)
similar_to_SR.head()

#矩阵中有许多缺失的值,将删除null值并将correlation results转化为dataframe
corr_FG = pd.DataFrame(similar_to_FG, columns=['Correlation'])
corr_FG.dropna(inplace=True)
corr_FG.head()
corr_SR = pd.DataFrame(similar_to_SR, columns=['Correlation'])
corr_SR.dropna(inplace=True)
corr_SR.head()

#有些电影被评价的次数很低,最终可能仅仅因为一两个人给了5分而被推荐 <= 设置阈值
corr_FG = corr_FG.join(rating['number_of_ratings'])
corr_SR = corr_SR.join(rating['number_of_ratings'])

#获取并查看前10部最为相关的电影(评论次数要大于30)
corr_FG[corr_FG['number_of_ratings'] > 30].sort_values(by='Correlation', ascending=False).head(10)
#自动编码器和受限的Boltzmann机器也常用于构建高级推荐系统