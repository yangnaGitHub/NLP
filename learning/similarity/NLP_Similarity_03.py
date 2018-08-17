# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 20:33:02 2018

@author: yangna

e-mail:ityangna0402@163.com
"""

import numpy as np
import os
import jieba
import copy

def Change_Workdir(workpath='E:\AboutStudy\code\python'):
    if not os.path.exists(workpath):
        os.makedirs(workpath)
    if os.getcwd() != workpath:
        os.chdir(workpath)

def Get_Tests_Dict(filename='data\\NLP_Similarity_00_01.txt'):
    with open(filename, encoding='utf8', mode='r') as fd:
        tests_dict = [test.strip('\n').split('\t') for test in fd.readlines()]
    return tests_dict

def Get_Train_Dict(filename='data\\NLP_Similarity_00_02.txt'):
    trains_dict_end = {}
    with open(filename, encoding='utf8', mode='r') as fd:
        trains_dict = dict([train.strip('\n').split('\t') for train in fd.readlines()])
        for key, vals in trains_dict.items():
            trains_dict_end[int(key)] = vals.split('&&&')
    return trains_dict_end

stop_words_list = []
def Get_Stop_Words(filename='data\\NLP_Similarity_00_03.txt'):
    with open(filename, encoding='utf8', mode='r') as fd:
        stop_words_list.extend([test.strip() for test in fd.readlines()])

#def Get_SortedWords_Score(words_len):
#    sortedwords_list = [index for index in range(words_len, 0, -1)]
#    sortedwords_sum = sum(sortedwords_list)
#    for index in range(1, words_len):
#        sortedwords_list[index] += sortedwords_list[index - 1]
#    return list(np.array(sortedwords_list) / sortedwords_sum)
#
#def Get_SortedWords_List(maxwordslen = 20):
#    sortedwords_list = [[0]]
#    for index in range(1, maxwordslen+1):
#        sortedwords_list.append(Get_SortedWords_Score(index))
#    return sortedwords_list

special_list = ['?', '？', '，', '.', '。', '、', ';', '：', '!', 'ˉ', '∶', ':', '"', '`', '·', '…', '—', '～', '-', '〃', '‘', '’', '“', '”', '〝', '〞']
def Cut_Strings(string):
    cut_list = list(jieba.cut(string, cut_all=False))
    return [word for word in [cut_word for cut_word in cut_list if cut_word not in special_list] if word not in stop_words_list]

#def Get_Test_Cut_List(tests_dict):
#    test_cut_list = []
#    for tests in tests_dict:
#        test_cut_list.append(Cut_Strings(tests[0]))
#    return test_cut_list
#
trains_cut_map = {}
def Get_Train_Cut_Dict(trains_dict):
    trains_cut_map = {}
    train_cut_list = []
    for index in range(1, len(trains_dict)+1):
        train_cut_list.clear()
        for train in trains_dict[index]:
            train_cut_list.append(Cut_Strings(train))
        trains_cut_map[index] = copy.deepcopy(train_cut_list)

trains_key_point_map = {}
def Calc_Train_Sets_Keypoint():
    for index in range(1, len(trains_cut_map)+1):
        trains_list = []
        for train in trains_cut_map[index]:
            if len(trains_list) == 0:
                trains_list = train
            else:
                trains_list = list(set(trains_list)&set(train))
        trains_key_point_map[index] = trains_list

def Get_Score(test_list, train_list):
    if test_list == train_list:
        return 1
    else:
        start_index  = min(len(test_list), len(train_list) - 1) if (len(train_list) > len(test_list)) else len(train_list)
        for index in range(start_index, 0, -1):
            tests_lists = [test_list[xindex:xindex+index] for xindex in range(0, len(test_list)-index+1)]
            train_lists = [train_list[xindex:xindex+index] for xindex in range(0, len(train_list)-index+1)]
            #去重
            if index == 1:
                no_tests_lists = []
                no_train_lists = []
                for tests in tests_lists:
                    if tests not in no_tests_lists:
                        no_tests_lists.append(tests)
                for train in train_lists:
                    if train not in no_train_lists:
                        no_train_lists.append(train)
                tests_lists = no_tests_lists
                train_lists = no_train_lists
            #高纬度匹配到更高的系数
            #只有一个词就做特殊处理
            one_count = [1 if tests==train else 0 for tests in tests_lists for train in train_lists].count(1)
            print(tests_lists)
            print(train_lists)
            print(index, one_count)
            if one_count != 0:
                return (2 * one_count) / (len(tests_lists) + len(train_lists))
        return 0

def Get_Score_Matrix(tests_dict, trains_dict):
    Score_Matrix = np.zeros((len(tests_dict), len(trains_dict)), dtype=float)
    for xindex, test in enumerate(tests_dict):
        current_test_list = Cut_Strings(test[0])
        for yindex in range(1, len(trains_dict)+1):
            Score = 0.0
            for index, train in enumerate(trains_dict[yindex]):
                current_train_list = trains_cut_map(yindex)[index]
                score = Get_Score(current_test_list, current_train_list)
                print('(%d,%d)(test)%s:(train)%s ==> (score)%f' % (xindex, yindex, test[0], train, score))
                Score += score
            Score_Matrix[xindex][yindex-1] = Score / len(trains_dict[yindex])
    return Score_Matrix

#trains_cut_map = {}
#def Get_Score_Matrix(tests_dict, trains_dict, sortedwords_list):
#    Score_Matrix = np.zeros((len(tests_dict), len(trains_dict)), dtype=float)
#    for xindex, test in enumerate(tests_dict):
#        current_test_list = Cut_Strings(test[0])
#        for yindex in range(1, len(trains_dict)+1):
#            score_list = []
#            for train in trains_dict[yindex]:
#                score = 0.0
#                if train in trains_cut_map:
#                    current_train_list = trains_cut_map[train]
#                else:
#                    current_train_list = Cut_Strings(train)
#                    trains_cut_map[train] = copy.deepcopy(current_train_list)
#                start_index = len(current_train_list)
#                current_test_index = [start_index] * len(current_test_list)
#                same_words = set(current_test_list)&set(current_train_list)
#                diff_words = (set(current_test_list)|set(current_train_list)) - same_words
#                for same_word in same_words:
#                    current_test_index[current_test_list.index(same_word)] = current_train_list.index(same_word)
#                sorted_lists = []
#                for current_index in current_test_index:
#                    if current_index < start_index:
#                        if sorted_lists:
#                            if current_index == (sorted_lists[-1][-1] + 1):
#                                sorted_lists[-1].append(current_index)
#                            else:
#                                sorted_lists.append([current_index])
#                        else:
#                            sorted_lists.append([current_index])
#                for sorted_list in sorted_lists:
#                    score += sortedwords_list[len(sorted_list)-1]#sortedwords_list[start_index][len(sorted_list)-1]
#                score -= minuswords_list[len(diff_words)-1]
#                print_string = ''
#                if int(test[1]) == yindex:
#                    print_string = 'true >>> ' 
#                print_string += '(%d,%d)(test)%s:(train)%s ==> (score)%f' % (xindex, yindex, test[0], train, score)
#                print(print_string)
#                score_list.append(score)
#            Score_Matrix[xindex][yindex-1] = sum(score_list) / len(score_list)
#    return Score_Matrix

def Get_Accuracy(tests_dict, cross_matrix, threshold=0.0, maxval=True, prefix='string'):
    label = [int(test[1]) for test in tests_dict]
    if maxval:
        if threshold == 0:
            threshold = -np.inf
        #real = [np.argmax(value)+1 if max(value) > threshold else 0 for value in cross_matrix]
        real = [np.argmax(value)+1 for value in cross_matrix]
    else:
        if threshold == 0:
            threshold = np.inf
        real = [np.argmin(value)+1 if min(value) < threshold else 0 for value in cross_matrix]
    #compare = list(map(lambda x, y: 1 if x==y else 0, label, real))
    #print([(real[index], label[index]) for index in compare if index==0])
    for index, tag in enumerate(label):
        if tag != real[index]:
            print(index, real[index], tag)
    #print('(%s)Accuracy = %f' % (prefix, compare.count(1)/len(compare)))

#概率密度函数
#一句话中存在相同的词
if __name__ == '__main__':
    Change_Workdir()
    tests_dict = Get_Tests_Dict()
    trains_dict = Get_Train_Dict()
    Get_Stop_Words()
    Score_Matrix = Get_Score_Matrix(tests_dict, trains_dict)
#    sortedwords_list = Get_SortedWords_List()
#    sortedwords_list = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
#    minuswords_list = [1, 1, 2, 2, 3, 3, 5, 5, 8, 8, 13, 13, 21, 21, 34, 34, 55, 55, 89, 89, 144,  144, 233, 233, 377, 377, 610, 610, 987, 987, 1597, 1597]
#    Score_Matrix = Get_Score_Matrix(tests_dict, trains_dict, sortedwords_list)
    max_lists = [max(value) for value in Score_Matrix]
    print(min(max_lists), max(max_lists))
    Get_Accuracy(tests_dict, Score_Matrix, prefix='sorting words %f' % (200/100), threshold=0)
    for xindex in range(14, 84, 2):#0.1-0.7
        Get_Accuracy(tests_dict, Score_Matrix, prefix='sorting words %f' % (xindex/100), threshold=xindex/100)
    #Get_Accuracy(tests_dict, Score_Matrix, prefix='full string %f' % 0.0, threshold=1)