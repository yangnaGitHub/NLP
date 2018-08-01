# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:13:59 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import os
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import metrics

class TFIDF(object):
    """
    TFIDF for text classfication
    """
    def __init__(self, args, params, log):
        self.args = args
        self.params = params
        self.log = log
        self.construct()
    
    def get_option(self, section, option, wclass='str'):
        if 'str' == wclass:
            return self.params.get(section, option)
        elif 'bool' == wclass:
            return self.params.getboolean(section, option)
        elif 'int' == wclass:
            return self.params.getint(section, option)
        elif 'float' == wclass:
            return self.params.getfloat(section, option)

    def print_log(self, message):
        if self.args.local_debug:
            print('[%s][%s:%s]' % (os.path.basename(__file__), self.__class__.__name__, traceback.extract_stack()[-2][2]), message)
        if self.args.print_to_log:
            if self.log:
                self.log.print_to_file(message)
    
    def predict(self, x_):
        features = self.feature.transform(x_)
        pre = self.model.predict(features)
        return pre

    def metrics_result(self, actual, predict):
        accuracy = metrics.precision_score(actual, predict, average='weighted')
        recall = metrics.recall_score(actual, predict, average='weighted')
        f1_score = metrics.f1_score(actual, predict, average='weighted')
        return accuracy, recall, f1_score
    
    def construct(self):
        '''
        一般有3个步骤
        1>load数据
        2>提取特征
         TFIDF
        3>分类
         SVM + NB...
        CountVectorizer只考虑每种词汇在该训练文本中出现的频率
        TfidfVectorizer除了考量某一词汇在当前训练文本中出现的频率之外,同时关注包含这个词汇的其它训练文本数目的倒数
         #自定义一个停用词表,如果不指定停用词表,则默认将所有单个汉字视为停用词,但可以设token_pattern=r"(?u)\b\w+\b",即不考虑停用词
        '''
        self.feature = TfidfVectorizer(token_pattern=r'(?u)\b\w\w+\b', max_df=self.get_option('summary', 'max_df', 'float'))
        method = self.get_option('summary', 'classfication', 'str')
        if 'SVM' == method:
            self.model = SVC(C=self.get_option('SVM', 'C_factor', 'float'), kernel=self.get_option('SVM', 'kernel', 'str'))
        elif 'NB' == method:
            self.model = MultinomialNB(alpha=self.get_option('NB', 'alpha', 'float'))

    def process(self, train_x, train_y):
        train_features = self.feature.fit_transform(train_x)
        self.model.fit(train_features, train_y)
    
class MyTHIDF(object):
    """
    TFIDF(my own code) for text classfication
    """
    def __init__(self, args, params):
        os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
        self.args = args
        self.params = params
        self.construct()
        
    def get_option(self, section, option, wclass='str'):
        if 'str' == wclass:
            return self.params.get(section, option)
        elif 'bool' == wclass:
            return self.params.getboolean(section, option)
        elif 'int' == wclass:
            return self.params.getint(section, option)
        elif 'float' == wclass:
            return self.params.getfloat(section, option)

    def print_log(self, message):
        if self.args.local_debug:
            print('[%s][%s:%s]' % (os.path.basename(__file__), self.__class__.__name__, traceback.extract_stack()[-2][2]), message)
        if self.args.print_to_log:
            pass
    def construct(self):
        pass