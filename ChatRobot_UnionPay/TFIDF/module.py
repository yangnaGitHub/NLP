# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:13:59 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import os
import traceback
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import metrics
import time
from datetime import timedelta
from collections import defaultdict
import pandas as pd
import numpy as np

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
    
    def get_quest_tfidf(self, quests):
        #return self.feature.transform(quests)
        return self.feature.transform(self.vectorizer.transform(quests))
    
#    def use_gpu_calcScore(self, originals, des):
#        import pycuda.autoinit
#        import pycuda.driver as drv
#        from pycuda.compiler import SourceModule
#        mod = SourceModule("""
#                           __global__ void calc_them(float *dest, float *a, float *b)
#                           {
#                           const int i = threadIdx.x;
#                           dest[i] = a[i] * b[i];
#                           }
#                           """)
#        calc_them = mod.get_function("calc_them")
#        
#        originals = self.get_quest_tfidf(originals).toarray()
#        des = des.transpose()
#        des_abs = np.linalg.norm(des)
#        
#        calc_them(drv.Out(dest), drv.In(a), drv.In(b), block=(400,1,1), grid=(1,1))
    
    def calcScore(self, originals, des):
        originals = self.get_quest_tfidf(originals).toarray()
        des = des.transpose()
        des_abs = np.linalg.norm(des)
        score = 0
        for ori in originals:
#            try:
#                score += np.dot(des, ori) / np.linalg.norm(ori)*des_abs
#            except RuntimeWarning:
#                self.print_log('calcScore:{}:{}+{}'.format(np.linalg.norm(ori)*des_abs, ori, des))
#                score += 0
            if 0 == np.linalg.norm(ori)*des_abs:
                self.print_log('calcScore:{}:{}+{}'.format(np.linalg.norm(ori)*des_abs, ori, des))
                score += 0
            else:
                score += np.dot(des, ori) / np.linalg.norm(ori)*des_abs
        return score/originals.shape[0]

    #pandas
    #      question1
    #type1
    #
    def similarity(self, features):
        if hasattr(self, 'saveData'):
            print_per_batch = self.get_option('summary', 'print_per_batch', 'int')
            start_time = time.time()
            total_batch = 0
            labels = self.saveData.keys()
            score = {'labels':list(labels)}
            for index in range(features.shape[0]):
                score['test_{}'.format(index)] = [0.0] * len(labels)
            score = pd.DataFrame(score)
            for vindex, feature in enumerate(features.toarray()):
                for hindex, label in enumerate(labels):
                    score.loc[hindex, 'test_{}'.format(vindex)] = self.calcScore(self.saveData[label], feature)
                    total_batch += 1
                    if total_batch % print_per_batch == 0:
                        time_dif = self.get_time_dif(start_time)
                        self.print_log('total_batch: {0}, Train Loss: {1}, Time: {2}'.format(total_batch, score[1:2], time_dif)) 
            #self.print_log('{}'.format(score))
            return [list(labels)[score['test_{}'.format(index)].idxmax()] for index in range(features.shape[0])]
        else:
            self.print_log('there not exist original question')
            return None
    
    def predict(self, x_):
        features = self.get_quest_tfidf(x_)
        #print(x_, features)
        #self.print_tfidf(features)
        method = self.get_option('summary', 'classfication', 'str')
        if method in ['SVM', 'NB']:
            return self.model.predict(features)
        elif method in ['similarity']:
            return self.similarity(features)
    
    def print_tfidf(self, features):
        word = self.vectorizer.get_feature_names()#获取词袋模型中的所有词语
        weight = features.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
        for index in range(len(weight)):    
            for jndex in range(len(word)):
                print(index, jndex, word[jndex], weight[index][jndex])
        
    def metrics_result(self, actual, predict):
        accuracy = metrics.precision_score(actual, predict, average='weighted')
        recall = metrics.recall_score(actual, predict, average='weighted')
        f1_score = metrics.f1_score(actual, predict, average='weighted')
        return accuracy, recall, f1_score

    def get_time_dif(self, start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))
    
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
        #from sklearn.feature_extraction.text import TfidfVectorizer
        #self.feature = TfidfVectorizer(token_pattern=r'(?u)\b\w\w+\b', max_df=self.get_option('summary', 'max_df', 'float'))#向量化等预处理

        from sklearn.feature_extraction.text import TfidfTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        self.vectorizer = CountVectorizer()#该类会将文本中的词语转换为词频矩阵,矩阵元素a[i][j]表示j词在i类文本下的词频
        self.feature = TfidfTransformer()#该类会统计每个词语的tf-idf权值

        method = self.get_option('summary', 'classfication', 'str')
        if 'SVM' == method:
            self.model = SVC(C=self.get_option('SVM', 'C_factor', 'float'), kernel=self.get_option('SVM', 'kernel', 'str'))
        elif 'NB' == method:
            self.model = MultinomialNB(alpha=self.get_option('NB', 'alpha', 'float'))

    def process(self, train_x, train_y):
        #train_features = self.feature.fit_transform(train_x)
        #print(self.feature.get_feature_names())
        start_time = time.time()
        train_features = self.feature.fit_transform(self.vectorizer.fit_transform(train_x))#第一个fit_transform是计算tf-idf,第二个fit_transform是将文本转为词频矩阵
        time_dif = self.get_time_dif(start_time)
        self.print_log('train_features_Time:{0}'.format(time_dif))
        #self.print_tfidf(train_features)
        method = self.get_option('summary', 'classfication', 'str')
        if method in ['SVM', 'NB']:
            self.model.fit(train_features, train_y)
        elif method in ['similarity']:
            if not hasattr(self, 'saveData'):
                self.saveData = defaultdict(set)
            for save_x, save_y in zip(train_x, train_y):
                self.saveData[save_y].add(save_x)
        time_dif = self.get_time_dif(start_time)
        self.print_log('model_train_Time:{0}'.format(time_dif))

#版本00,最原始的TFIDF
def tfidf_00(docs):
    #init table(save freq)
    import math
    word_map = {}
    for index, doc in enumerate(docs):
        word_map[index] = {}
        for word in doc.split(' '):
            if '' == word:
                continue
            if word not in word_map[index]:
                word_map[index][word] = [1]
            else:
                word_map[index][word][0] += 1
    
    #tf value
    totaldoc = len(word_map)
    for docindex in range(totaldoc):
        doctotalwords = sum([freq[0] for freq in word_map[docindex].values()])
        for freq in word_map[docindex].values():
            doctfwords = freq[0]/doctotalwords
            freq.append(doctfwords)
    
    #idf value
    for docindex in range(totaldoc):
        for words in word_map[docindex].keys():
            existdoc = 0
            for key, value in word_map.items():
                if key == docindex:
                    continue
                if words in value.keys():
                    existdoc += 1
            word_map[docindex][words].append(math.log((totaldoc + 0)/(existdoc + 1)))#加1是为了平滑
    
    #calc tf-idf
    for docindex in range(totaldoc):
        for freq in word_map[docindex].values():
            freq.append(freq[1]*freq[2])
    
    return word_map

#版本01
def tfidf_01(docs):
    #init table(save freq)
    import math
    word_map = {}
#    vocab_map = {}
    for index, doc in enumerate(docs):
        word_map[index] = {}
        for word in doc.split(' '):
            if '' == word:
                continue
#            if word not in vocab_map:
#                vocab_map[word] = 1
#            else:
#                vocab_map[word] += 1
            
            if word not in word_map[index]:
                word_map[index][word] = [1]
            else:
                word_map[index][word][0] += 1
    
    #tf value
    totaldoc = len(word_map)
    doctotalwords_list = []
    for docindex in range(totaldoc):
        doctotalwords = sum([freq[0] for freq in word_map[docindex].values()])
        doctotalwords_list.append(doctotalwords)
        for freq in word_map[docindex].values():
            doctfwords = freq[0]/doctotalwords
            freq.append(doctfwords)
    
    #idf value
    calcdocwords = 1000
    for docindex in range(totaldoc):
        for words in word_map[docindex].keys():
            times = calcdocwords / doctotalwords_list[docindex]
            mine = word_map[docindex][words][0] * times
            existdoc = 0
            others = 0
            for key, value in word_map.items():
                if key == docindex:
                    continue
                if words in value.keys():
                    times = calcdocwords / doctotalwords_list[key]
                    others += value[words][0] * times
                    existdoc += 1
            ratio = mine / (mine + others)
            word_map[docindex][words].append(ratio * math.log((totaldoc + 0)/(existdoc + 1)))#加1是为了平滑
    
    #calc tf-idf
    for docindex in range(totaldoc):
        for freq in word_map[docindex].values():
            freq.append(freq[1]*freq[2])
    
    return word_map

#对于Q-A这种问题始终认为是不同问题
def tfidf_02(docs):
    pass

#from numba import vectorize
#@vectorize(["float32(float32, float32)"], target='cuda')
#def vectorAdd(a, b):
#    return a * b

#def use_gpu_calcScore(self, originals, des):
#    import pycuda.autoinit
#    import pycuda.driver as drv
#    from pycuda.compiler import SourceModule
#    mod = SourceModule("""
#                       __global__ void calc_them(float *dest, float *a, float *b)
#                       {
#                       const int i = threadIdx.x;
#                       dest[i] = a[i] * b[i];
#                       }
#                       """)
#    calc_them = mod.get_function("calc_them")
    
    #a = np.array([[1, 2, 3], [2, 2, 3], [3, 2, 3], [4, 2, 3]])
    #b = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
#    a = np.array([1, 2, 3, 4])
#    b = np.array([1, 2, 3, 4])
#    print(vectorAdd(a, b))
    

#if __name__ == '__main__':
#    import numpy as np
#    inputlist = np.array(['怎样在 Apple   ID 上 关联 银行卡','如何在 Apple   ID 上 关联 银行卡', '何如在 Apple   ID 上 关联 银行卡', '咋样在 Apple   ID 上 关联 银行卡'
#, '怎么样在 Apple   ID 上 关联 银行卡', '如何在 App   Store 中 绑 卡 ？'
#, '怎么在 App   Store 中 绑 卡 ？', '怎样在 App   Store 中 绑 卡 ？', '怎么在 Itunes 中 绑 卡 ？'
#, '如何在 Itunes 中 绑 卡 ？', '怎样在 Itunes 中 绑 卡 ？', '什么是APP STORE？'
#, 'App Store是干嘛的？', '什么是App Store?'])
#    word_map = tfidf_01(inputlist)