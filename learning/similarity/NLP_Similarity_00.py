# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 09:55:15 2018

@author: natasha_yang
"""

import numpy as np
import jieba.analyse
import gensim
import random

DEBUG_STRING = 0

def Get_Test_Sets(filename='NLP_S_Similarity_00_01.txt'):
     with open(filename, encoding='utf8', mode='r') as fd:
         test_sets = [data.replace('\n', '').split('\t') for data in fd.readlines()]
         return test_sets

def Get_Question_Sets(filename='NLP_S_Similarity_00_02.txt'):
    question_dicts = {}
    with open(filename, encoding='utf8', mode='r') as fd:
        question_sets = [data.replace('\n', '').split('\t') for data in fd.readlines()]
        for data in question_sets:
            question_dicts[int(data[0])] = data[-1].split('&&&')
        return question_dicts

HashTable = {}
class SmiHash():
    def __init__(self, src, des, hashbits=128):
        self.srcstr = src
        self.desstr = des
        self.hashbits = hashbits
        if DEBUG_STRING:
            print(self.srcstr, self.desstr)
    
    def Calc_SimHash_For_String(self):
        self.srcsimhashlist = self.Calc_SimHash_For_Each_String(self.srcstr)
        self.dessimhashlist = self.Calc_SimHash_For_Each_String(self.desstr)
        if DEBUG_STRING:
            print('srcsimhashlist', self.srcsimhashlist, self.dessimhashlist)
        return self.Similarity_For_SimHash()
    
    def Similarity_For_SimHash(self):
        #第一种方式
        self.srcsimhash = self.list_covert_float(self.srcsimhashlist)
        self.dessimhash = self.list_covert_float(self.dessimhashlist)
        #if DEBUG_STRING:
        #    print('srcsimhash', self.srcsimhash, self.dessimhash)
        #    print(min(self.srcsimhash, self.dessimhash), max(self.srcsimhash, self.dessimhash))
        #return min(self.srcsimhash, self.dessimhash) / max(self.srcsimhash, self.dessimhash)
        #海明距离
        xorvalue = (self.srcsimhash ^ self.dessimhash) & ((1 << self.hashbits) - 1)
        notone = 0
        while xorvalue:
            notone += 1
            xorvalue &= xorvalue-1
        return notone  
    
    def list_covert_float(self, hashlist):
        simhashvalue = 0
        for index in range(self.hashbits):
            simhashvalue += hashlist[index] * (1 << index)
        return simhashvalue
    
    def Calc_SimHash_For_Each_String(self, string):
        stringHash = np.zeros((self.hashbits, ), dtype=float)
        hashbits = [0.0] * self.hashbits
        stringHashDict = {}
        #weight要归一化么？
        for words, weight in jieba.analyse.extract_tags(string, withWeight=True):
            if words not in HashTable:
                hashValue = self.words_hash(words)
                if DEBUG_STRING:
                    print('hashValue', bin(hashValue))
                for index in range(self.hashbits):
                    hashbits[index] = 1 if hashValue & (1 << index) else -1
                HashTable[words] = hashbits
            hashbits = [bit * weight for bit in HashTable[words]]
            if DEBUG_STRING:
                print('hashbits', words, HashTable[words], hashbits, weight)
            stringHashDict[words] = np.array(hashbits)
        for Value in stringHashDict.values():
            stringHash += Value
            if DEBUG_STRING:
                print('stringHash', stringHash, Value)
        return [1 if Hashbit > 0 else 0 for Hashbit in stringHash]
    
    def words_hash(self, words):
        if words == "":
            return 0
        else:
            x = ord(words[0]) << 7
            m = 1000003
            mask = 2 ** self.hashbits - 1
            for c in words:
                x = ((x * m) ^ ord(c)) & mask
            x ^= len(words)
            if x == -1:
                x = -2
            return x

#从文字上面来说
def Calc_Each_SimHash(teststr, questionstr):
    simhash = SmiHash(teststr, questionstr)
    return simhash.Calc_SimHash_For_String()

def Calc_SimHash(teststr, questions):
    SmiHashValue = 0
    #相似度求平均
    for index in range(len(questions)):
        SmiHashValue += Calc_Each_SimHash(teststr, questions[index])
    return SmiHashValue / len(questions)

#从词向量上面来说
def Load_Word2Vec_Model(modelname='model\\word2vec\\news_12g_baidubaike_20g_novel_90g_embedding_64.bin'):
    return gensim.models.KeyedVectors.load_word2vec_format(modelname, binary=True)
model = Load_Word2Vec_Model()

SematicWordsTable = {}
SematicTable = {}
#StringVector = {}#保存句子词向量
#skipwords = ['?', ',', '"', '？', '，']#跳过特殊字符，没用
class Sematic:
    def __init__(self, src, des, model, mu=0, sigma=5):
        self.srcstr = src
        self.desstr = des
        self.model = model
        self.vectorlen = len(model['北京'])
        self.mu = mu
        self.sigma = sigma
        
    def Calc_Sematic_For_String(self):
        self.srcwordslist = list(jieba.cut(self.srcstr, cut_all=False))
        self.deswordslist = list(jieba.cut(self.desstr, cut_all=False))
        #reduce(lambda skipword: self.srcwordslist.remove(skipword), skipwords)
        #reduce(lambda skipword: self.deswordslist.remove(skipword), skipwords)
        self.vectorrows = max(len(self.srcwordslist), len(self.deswordslist))#paddding
        #print(self.vectorrows, self.srcwordslist, self.deswordslist)
        self.srcvec = self.Calc_Vector_For_String(self.srcwordslist)
        self.desvec = self.Calc_Vector_For_String(self.deswordslist)
        #print(list(self.srcwordslist), self.srcvec)
        #print(list(self.deswordslist), self.desvec)
        #计算词向量矩阵距离
        vec = self.srcvec - self.desvec
        return np.trace(np.dot(vec, vec.transpose())) ** 0.5
    
    def Calc_Vector_For_String(self, wordslist):
        wordsvec = np.zeros((self.vectorrows, self.vectorlen), dtype=float)
        for index, words in enumerate(wordslist):
            try:
                wordsvec[index] = self.model[words]
            except KeyError:
                if words not in SematicTable:
                    if words not in SematicWordsTable:
                        print('please update sematic table for %s' % words)
                        wordsvec[index] = np.array([random.gauss(self.mu, self.sigma) for i in range(self.vectorlen)])
                    else:
                        for index, similarity in enumerate(SematicWordsTable[words]):
                            try:
                                wordsvec[index] += np.array(self.model[SematicWordsTable[similarity]])
                            except KeyError:
                                 wordsvec[index] += np.array([random.gauss(self.mu, self.sigma) for i in range(self.vectorlen)])
                        wordsvec[index] /= (index + 1)
                    SematicTable[words] = wordsvec[index]
                else:
                    wordsvec[index] = SematicTable[words]
        return wordsvec

def Calc_Each_Sematic(teststr, questionstr):
    sematic = Sematic(teststr, questionstr, model)
    return sematic.Calc_Sematic_For_String()
    
def Calc_Sematic(teststr, questions):
    SematicValue = 0
    #词向量求平均
    #可不可以词向量矩阵的聚类/LDA(TF-IDF)？？（类中保存词向量==>不同维度怎么求聚类）
    for index in range(len(questions)):
        SematicValue += Calc_Each_Sematic(teststr, questions[index])
    return SematicValue / len(questions)

#从语义上来说
#基于字典，基于大数据的统计，数据不够

def Get_Similarity_Matrix(test_sets, question_dicts):
    SmiHashMatrix = np.zeros((len(test_sets), len(question_dicts)), dtype=float)
    SemanticMatrix = np.zeros((len(test_sets), len(question_dicts)), dtype=float)
    for xindex, tests in enumerate(test_sets):
        for yindex in range(1, len(question_dicts)+1):
            SmiHashMatrix[xindex][yindex-1] = Calc_SimHash(tests[0], question_dicts[yindex])
            SemanticMatrix[xindex][yindex-1] = Calc_Sematic(tests[0], question_dicts[yindex])
    return SmiHashMatrix, SemanticMatrix

if __name__ == '__main__':
    test_sets = Get_Test_Sets()
    question_dicts = Get_Question_Sets()
    SematicWordsTable['DNA'] = ['染色体', '遗传', '基因', '螺旋体', '化合物']
    SmiHashMatrix, SemanticMatrix = Get_Similarity_Matrix(test_sets, question_dicts)
    
    if DEBUG_STRING:
        import matplotlib.pyplot as plt
        Smimintest = [min(SmiHash) for SmiHash in SmiHashMatrix]
        Semmintest = [min(Semantic) for Semantic in SemanticMatrix]
        des_test = [int(tests[1]) for tests in test_sets]
        plt.subplot(311)
        plt.plot(range(1, len(Smimintest)+1), Smimintest)
        plt.subplot(312)
        plt.plot(range(1, len(Semmintest)+1), Semmintest)
        plt.subplot(313)
        plt.plot(range(1, len(Semmintest)+1), [1 if data > 0 else 0 for data in des_test])
        plt.show()
        #测试阈值20也还有35个错误
        for xindex in range(15, 31):
            test = [np.argmin(SmiHash) if min(SmiHash) < xindex else -1 for SmiHash in SmiHashMatrix]
            count = 0
            for index in range(len(test)):
                if (test[index]+1) != des_test[index]:
                    count += 1
            print(xindex, count)
        for xindex in range(100, 160):
            test = [np.argmin(Semantic) if min(Semantic) < xindex else -1 for Semantic in SemanticMatrix]
            count = 0
            for index in range(len(test)):
                if (test[index]+1) != des_test[index]:
                    count += 1
            print(xindex, count)
            
    SmiHashResults = np.array([np.argmin(SmiHash) if min(SmiHash) < 20 else -1 for SmiHash in SmiHashMatrix]) + 1#针对0的处理
    SemanticResults = np.array([np.argmin(Semantic) if min(Semantic) < 125 else -1 for Semantic in SemanticMatrix]) + 1#针对0的处理
    Results = np.array([int(tests[1]) for tests in test_sets])
    count = 0
    for index, Result in enumerate(Results):
        if Result != SmiHashResults[index] and Result != SemanticResults[index]:
            print('index %s similarity error, (%d: %d(SmiHash) %d(Vector))' % (index, Result, SmiHashResults[index], SemanticResults[index]))
            count += 1
    print('accuracy = %f' % (1 - (count / len(test_sets))))#精确度太低73%，如何提高精确度