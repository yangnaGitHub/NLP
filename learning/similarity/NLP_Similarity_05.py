# -*- coding: utf-8 -*-
"""
Created on Tue May  1 21:12:36 2018

@author: yangna

e-mail:ityangna0402@163.com
"""
import numpy as np
import os
import copy
import jieba
import collections
#如何分词
#句子之间求交集
#引入权重和优先级的概念
#匹配原则
def Change_Workdir(workpath='F:\\yangna\\code\\python&r'):
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

special_list = ['?', '？', '，', '.', '。', '、', ';', '：', '!', 'ˉ', '∶', ':', '"', '`', '·', '…', '—', '～', '-', '〃', '‘', '’', '“', '”', '〝', '〞']
stop_words_list = []
def Get_Stop_Words(filename='data\\NLP_Similarity_00_03.txt'):
    with open(filename, encoding='utf8', mode='r') as fd:
        stop_words_list.extend([test.strip() for test in fd.readlines()])

def Cut_Strings(string):
    cut_list = list(jieba.cut(string, cut_all=False))
    return [word for word in [cut_word for cut_word in cut_list if cut_word not in special_list] if word not in stop_words_list]

trains_cut_map = {}
def Get_Train_Cut_Dict(trains_dict):
    if len(trains_cut_map) != 0:
        trains_cut_map.clear()
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

def Get_Cross_Matrix(tests_dict, trains_dict, method='default'):
    cross_matrix = np.zeros((len(tests_dict), len(trains_dict)), dtype=float)
    for xindex, test in enumerate(tests_dict):
        test_set = set(Cut_Strings(test[0]))
        for yindex in range(1, len(trains_dict)+1):
            count = 0.0
            for index, train in enumerate(trains_dict[yindex]):
                train_set = set(trains_cut_map[yindex][index])
                if method == 'default':
                    count += len(train_set&test_set) / len(train_set|test_set)
                elif method == 'Add_Weight':
                    same_lists = [1 if same in trains_key_point_map[yindex] else 0 for same in (train_set&test_set)]
                    count += ((0.5*same_lists.count(1)) + (0.1*same_lists.count(0)))
            cross_matrix[xindex][yindex-1] = count / len(trains_dict[yindex])
    return cross_matrix

def Get_Score_Matrix(tests_dict, trains_dict, sortedwords_list, minuswords_list):
    Score_Matrix = np.zeros((len(tests_dict), len(trains_dict)), dtype=float)
    for xindex, test in enumerate(tests_dict):
        current_test_list = Cut_Strings(test[0])
        for yindex in range(1, len(trains_dict)+1):
            score_list = []
            for index, train in enumerate(trains_dict[yindex]):
                score = 0.0
                current_train_list = trains_cut_map[yindex][index]
                start_index = len(current_train_list)
                current_test_index = [start_index] * len(current_test_list)
                same_words = set(current_test_list)&set(current_train_list)
                diff_words = (set(current_test_list)|set(current_train_list)) - same_words
                for same_word in same_words:
                    current_test_index[current_test_list.index(same_word)] = current_train_list.index(same_word)
                sorted_lists = []
                for current_index in current_test_index:
                    if current_index < start_index:
                        if sorted_lists:
                            if current_index == (sorted_lists[-1][-1] + 1):
                                sorted_lists[-1].append(current_index)
                            else:
                                sorted_lists.append([current_index])
                        else:
                            sorted_lists.append([current_index])
                for sorted_list in sorted_lists:
                    score += sortedwords_list[len(sorted_list)-1]#sortedwords_list[start_index][len(sorted_list)-1]
                score -= minuswords_list[len(diff_words)-1]
#                print_string = ''
#                if int(test[1]) == yindex:
#                    print_string = 'true >>> ' 
#                print_string += '(%d,%d)(test)%s:(train)%s ==> (score)%f' % (xindex, yindex, test[0], train, score)
#                print(print_string)
                score_list.append(score)
            Score_Matrix[xindex][yindex-1] = sum(score_list) / len(score_list)
    return Score_Matrix

def Get_Accuracy(tests_dict, cross_matrix, threshold=0.0, maxval=True, prefix='string'):
#    label = [int(test[1]) for test in tests_dict]
    if maxval:
        if threshold == 0:
            threshold = -np.inf
        real = [np.argmax(value)+1 if max(value) > threshold else 0 for value in cross_matrix]
        #real = [np.argmax(value)+1 for value in cross_matrix]
    else:
        if threshold == 0:
            threshold = np.inf
        real = [np.argmin(value)+1 if min(value) < threshold else 0 for value in cross_matrix]
#    compare = [1 if tag==real[index] else 0 for index, tag in enumerate(label)]
#    #compare = list(map(lambda x, y: 1 if x==y else 0, label, real))
#    #print([(index, real[index], label[index]) for index, tag in enumerate(label)  if compare[index] == 0 ])
#    print('(%s)Accuracy = %f' % (prefix, compare.count(1)/len(compare)))
    return real

if __name__ == '__main__':
    Change_Workdir()
    tests_dict = Get_Tests_Dict()
    trains_dict = Get_Train_Dict()
    #Get_Stop_Words()
    Get_Train_Cut_Dict(trains_dict)
    Calc_Train_Sets_Keypoint()
    
    Cross_Matrix = Get_Cross_Matrix(tests_dict, trains_dict)
    max_lists = [max(value) for value in Cross_Matrix]
#    print(min(max_lists), max(max_lists))
#    for xindex in range(20, 86, 2):#0.1-0.7
#        Get_Accuracy(tests_dict, Cross_Matrix, prefix='sets words %f' % (xindex/100), threshold=xindex/100)#0.30 73%
        
    sortedwords_list = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
    minuswords_list = [1, 1, 2, 2, 3, 3, 5, 5, 8, 8, 13, 13, 21, 21, 34, 34, 55, 55, 89, 89, 144,  144, 233, 233, 377, 377, 610, 610, 987, 987, 1597, 1597]
    Score_Matrix = Get_Score_Matrix(tests_dict, trains_dict, sortedwords_list, minuswords_list)
#    max_lists = [max(value) for value in Score_Matrix]
#    print(min(max_lists), max(max_lists))
#    for xindex in range(50, 300, 2):#0.1-0.7
#        Get_Accuracy(tests_dict, Score_Matrix, prefix='sorts words %f' % (xindex/100), threshold=xindex/100)#0.30 73%
    
    Score_Matrix_Remove_Stop = Get_Cross_Matrix(tests_dict, trains_dict, method='Add_Weight')
#    max_lists = [max(value) for value in Score_Matrix_Remove_Stop]
#    print(min(max_lists), max(max_lists))
#    for xindex in range(10, 100, 2):#0.1-0.7
#        Get_Accuracy(tests_dict, Score_Matrix_Remove_Stop, prefix='point words %f' % (xindex/100), threshold=xindex/100)#0.30 73%
    
    TOTAL = np.zeros((3, len(tests_dict)), dtype=float)
    TOTAL[0] = Get_Accuracy(tests_dict, Cross_Matrix, prefix='sets words %f' % (23/100), threshold=0.23)
    TOTAL[1] = Get_Accuracy(tests_dict, Score_Matrix, prefix='sorts words %f' % (80/100), threshold=0.8)
    TOTAL[2] = Get_Accuracy(tests_dict, Score_Matrix_Remove_Stop, prefix='point words %f' % (54/100), threshold=0.54)
    finally_lists =[]
    for index in range(len(tests_dict)):
        results = collections.Counter(TOTAL[:, index])
        maxsum = -1
        maxkey = 0
        for result in results:
            maxkey = result if results[result] > maxsum else maxkey
            maxsum = max(maxsum, results[result])
        finally_lists.append(int(maxkey))
    label = [int(test[1]) for test in tests_dict]
    compare = [1 if tag==finally_lists[index] else 0 for index, tag in enumerate(label)]
    #print([(index, finally_lists[index], label[index]) for index, tag in enumerate(label)  if compare[index] == 0 ])
    for xindex, test in enumerate(tests_dict):
        print('calc: ', test[0], finally_lists[xindex],)
        print('real: ', test[0], label[xindex])
    print('Accuracy = %f' % (compare.count(1)/len(compare)))