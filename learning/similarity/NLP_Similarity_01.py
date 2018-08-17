# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 21:31:57 2018

@author: yangna

e-mail:ityangna0402@163.com
"""

import numpy as np
import os
import copy
import random

#如何分词
#句子之间求交集
#引入权重和优先级的概念
#匹配原则
#采取停止词

def Change_Workdir(workpath='F:\\yangna\\code\\python&r'):
    if not os.path.exists(workpath):
        os.makedirs(workpath)
    if os.getcwd() != workpath:
        os.chdir(workpath)

def Get_Words_Dict(filename='data\\words.txt'):
    with open(filename, encoding='utf8', mode='r') as fd:
        words_list = fd.readlines()[0].split('\t')
    return words_list

def Get_Tests_Dict(filename='data\\NLP_Similarity_00_01.txt'):
    with open(filename, encoding='utf8', mode='r') as fd:
        tests_dict = [test.strip('\n').split('\t') for test in fd.readlines()]
    return tests_dict

def Get_Train_Dict(filename='data\\NLP_Similarity_00_02.txt'):
    trains_dict = {}
    with open(filename, encoding='utf8', mode='r') as fd:
        trains_sets = [data.replace('\n', '').split('\t') for data in fd.readlines()]
        for data in trains_sets:
            trains_dict[int(data[0])] = data[-1].split('&&&')
    return trains_dict

def choice_str(words_len, str_list, words_list, string, index, s_size):
    word_len = random.choice(words_len)
    words_len.remove(word_len)
    if index+word_len-1 < s_size and string[index:index+word_len] in words_list:
        str_list.pop()
        str_list.append(string[index:index+word_len])
        index += (word_len - 1)
    return index

def choice_str_res(words_len, str_list, words_list, string, index, s_size):
    word_len = random.choice(words_len)
    words_len.remove(word_len)
    if index-word_len+1 > 0 and string[index-word_len:index] in words_list:
        str_list.pop()
        str_list.append(string[index-word_len:index])
        index -= (word_len - 1)
    return index

def Cut_String(string, words_list, weight=False):
    str_list = []
    str_list_res = []
    index = 0
    while True:
        if index >= len(string):
            break
        str_list.append(string[index])
        #增加随机性,可使用权重选择
        words_len = [2, 3, 4]
        while len(words_len) != 0:
            new_index = choice_str(words_len, str_list, words_list, string, index, len(string))
            if new_index != index:
                index = new_index
                break
        index += 1
        #优先匹配2-gram/3-gram/4-gram
#        if index+1 < len(string) and string[index:index+2] in words_list:
#            str_list.pop()
#            str_list.append(string[index:index+2])
#            index += 1
#        else:
#            if index+2 < len(string) and string[index:index+3] in words_list:
#                str_list.pop()
#                str_list.append(string[index:index+3])
#                index += 2
#            else:
#                if index+3 < len(string) and string[index:index+4] in words_list:
#                    str_list.pop()
#                    str_list.append(string[index:index+4])
#                    index += 3
        
    index = len(string)
    while True:
        if index <= 0:
            break
        str_list_res.append(string[index - 1])
        words_len = [2, 3, 4]
        while len(words_len) != 0:
            new_index = choice_str_res(words_len, str_list_res, words_list, string, index, len(string))
            if new_index != index:
                index = new_index
                break
        index -= 1
#        if index-1 > 0 and string[index-2:index] in words_list:
#            str_list_res.pop()
#            str_list_res.append(string[index-2:index])
#            index -= 1
#        else:
#            if index-2 > 0 and string[index-3:index] in words_list:
#                str_list_res.pop()
#                str_list_res.append(string[index-3:index])
#                index -= 2
#            else:
#                if index-3 > 0 and string[index-4:index] in words_list:
#                    str_list_res.pop()
#                    str_list_res.append(string[index-4:index])
#                    index -= 3
#        index -= 1
    diff_set = set(str_list)-set(str_list_res)#Notice:集合解决了匹配的问题,要用C/C++实现一个集合所有的功能
    if diff_set:
        cut_list = str_list
#        cut_words = ['不', '要', '还']
#        if 0 != sum([1 if cut_word in (set(str_list_res) - set(str_list)) else 0 for cut_word in cut_words]):
#            cut_list = str_list_res
        #正向分词和反向分词的冲突处理
        #print('diff: ', str_list, str_list_res)
        #diff:  ['不准', '确'] ['准确', '不'] ['准确', '不']
        #diff:  ['有意', '义'] ['意义', '有'] ['意义', '有']
        #diff:  ['还需', '要'] ['需要', '还'] ['需要', '还']
        #diff:  ['不适', '时'] ['适时', '不'] ['不适', '时']
        #diff:  ['最好', '的'] ['好的', '最'] ['最好', '的']
        #diff:  ['每个', '人'] ['个人', '每'] ['个人', '每']
        #diff:  ['谁都', '能'] ['都能', '谁'] ['都能', '谁']
        #diff:  ['一共', '有'] ['共有', '一'] ['一共', '有'] 
    else:
        cut_list = str_list
    if weight:
        #赋值权重,根据什么分权重呢,要根据语法,找动词什么的
        #如何知道一句话的关键点在哪儿
        #人类是怎么GET到一句话的关键点的?是怎么明白和理解语言的意思的
        #想象着学习英语
         #先学习字母表
         #在学习音标
         #学习简单的单词
         #造简单的句子
         #造不出句子的时候要学习语法
         #大量的记单词
          #语感??
        pass
    
    return cut_list    

def Cut_Trains(trains_dict, words_list, weight=False):
    back_trains_dict = copy.deepcopy(trains_dict)
    for key, vals in back_trains_dict.items():
        for index, val in enumerate(vals):
            vals.insert(index, Cut_String(val, words_list, weight))
            vals.remove(val)
        back_trains_dict[key] = vals
    return back_trains_dict

def Cut_Tests(tests_dict, words_list, weight=False):
    back_tests_dict = copy.deepcopy(tests_dict)
    for index, test in enumerate(back_tests_dict):
        test.insert(1, Cut_String(test[0], words_list, weight))
        test.pop(0)
    return back_tests_dict

def Get_Cross_Matrix(tests_dict, trains_dict, weight=False):
    cross_matrix = np.zeros((len(tests_dict), len(trains_dict)), dtype=float)
    for xindex, test in enumerate(tests_dict):
        test_set = set(test[0])
        for yindex in range(1, len(trains_dict)+1):
            count = 0.0
            for train in trains_dict[yindex]:
                train_set = set(train)
                count += len(train_set&test_set) / len(train_set|test_set)
            cross_matrix[xindex][yindex-1] = count / len(trains_dict[yindex])
    return cross_matrix

def Get_Accuracy(tests_dict, cross_matrix, threshold=0.0, maxval=True, prefix='string'):
    label = [int(test[1]) for test in tests_dict]
    if maxval:
        if threshold == 0:
            threshold = -np.inf
        real = [np.argmax(value)+1 if max(value) > threshold else 0 for value in cross_matrix]
    else:
        if threshold == 0:
            threshold = np.inf
        real = [np.argmin(value)+1 if min(value) < threshold else 0 for value in cross_matrix]
    compare = list(map(lambda x, y: 1 if x==y else 0, label, real))
    print('(%s)Accuracy = %f' % (prefix, compare.count(1)/len(compare)))
        
if __name__ == '__main__':
    Change_Workdir()
    words_list = Get_Words_Dict()
    tests_dict = Get_Tests_Dict()
    trains_dict = Get_Train_Dict()
    #求整个句子的集合交集
    cross_matrix = Get_Cross_Matrix(tests_dict, trains_dict)
    for xindex in range(10, 70, 2):#0.1-0.7
        Get_Accuracy(tests_dict, cross_matrix, prefix='full string %f' % (xindex/100), threshold=xindex/100)#0.35 80%
    #求分词的集合交集
    weight = False
    tests_dict_cut = Cut_Tests(tests_dict, words_list, weight)
    trains_dict_cut = Cut_Trains(trains_dict, words_list, weight)
    cross_matrix_cut = Get_Cross_Matrix(tests_dict_cut, trains_dict_cut, weight)
    for xindex in range(0, 70, 2):#0.1-0.7
        Get_Accuracy(tests_dict, cross_matrix_cut, prefix='cut string %f' % (xindex/100), threshold=xindex/100)#0.30 73%
    #赋予权重
#    weight = True
#    tests_dict_weight = Cut_Tests(tests_dict, words_list, weight)
#    trains_dict_weight = Cut_Trains(trains_dict, words_list, weight)
#    cross_matrix_weight = Get_Cross_Matrix(tests_dict_weight, trains_dict_weight, weight)
#    for xindex in range(0, 70, 2):#0.1-0.7
#        Get_Accuracy(tests_dict, cross_matrix_weight, prefix='cut and weight string %f' % (xindex/100), threshold=xindex/100)