# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 17:34:17 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

from TFIDF.module import TFIDF as TFIDF
from datetime import timedelta
import os
import traceback
import time
import configparser
import pickle
import numpy as np
#from sklearn.externals import joblib

class TFIDFOp():
    def __init__(self, args, data, log):
        self.module_name = 'TFIDF'
        self.model = None
        self.session = None
        self.args = args
        self.data = data
        self.log = log
        self.params = configparser.ConfigParser()
        if self.args.isoptionexist('use_model', 'module_conf'):
            module_conf = self.args.get_option('use_model', 'module_conf', 'str')
        else:
            module_conf = 'module.cfg'
        self.params.read('./' + self.module_name + '/' + module_conf)
        self.module_path = self.get_option('summary', 'module_path', 'str')
        self.pickle_file = self.module_path + '/' + self.get_option('summary', 'pickle_file', 'str')
    
    def print_log(self, message):
        if self.args.local_debug:
            print('[%s][%s:%s]' % (os.path.basename(__file__), self.__class__.__name__, traceback.extract_stack()[-2][2]), message)
        if self.args.print_to_log:
            if self.log:
                self.log.print_to_file(message)
    
    def get_option(self, section, option, wclass='str'):
        if 'str' == wclass:
            return self.params.get(section, option)
        elif 'bool' == wclass:
            return self.params.getboolean(section, option)
        elif 'int' == wclass:
            return self.params.getint(section, option)
        elif 'float' == wclass:
            return self.params.getfloat(section, option)
        
    def get_time_dif(self, start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))
    
    def reconstruct_input(self, source, combine=False):
        if combine:     
            reconstruct_list = []
            for quests in list(source.values()):
                string = ''
                for quest in quests:
                    string += quest
                    string += ' '
                string.strip()
                reconstruct_list.append(string)
            return np.array(reconstruct_list), np.array(list(source.keys()))
        else:
            return np.array(list(source.keys())), np.array(list(source.values()))
    
    def train(self):
        self.tfidf = TFIDF(self.args, self.params, self.log)
        
        start_time = time.time()
        
        combine = self.get_option('summary', 'combine_quest', 'bool')
        if combine:
            train_source = self.data.label_quest
            #test_source = self.data.label_testquest
        else:
            train_source = self.data.quest_label
        
        
        batch_x, batch_y = self.reconstruct_input(train_source, combine=combine)
        self.print_log('batch_x:{0}, batch_y:{1}'.format(batch_x.shape, batch_y.shape))
        self.tfidf.process(batch_x, batch_y)
        
        test_source = self.data.testquest_label
        test_x, test_y = self.reconstruct_input(test_source)
        self.print_log('test_x:{0}, test_y:{1}'.format(test_x.shape, test_y.shape))
        predict = self.tfidf.predict(test_x)
        try:
            accuracy, recall, f1_score = self.tfidf.metrics_result(test_y, predict)
        except Exception as e:
            self.print_log(e)
        self.conuter = self.tfidf.vectorizer
        self.feature = self.tfidf.feature
        method = self.get_option('summary', 'classfication', 'str')
        if method in ['SVM', 'NB']:
            self.model = self.tfidf.model
        time_dif = self.get_time_dif(start_time)
        self.print_log('精度:{0:>6.2}, 召回:{1:>6.2}, f1-score:{2:>6.2}, Time:{3}'.format(accuracy, recall, f1_score, time_dif))

        if not os.path.exists(self.module_path):
            os.makedirs(self.module_path)
#        joblib.dump(self.conuter, self.module_path+'/'+self.get_option('summary', 'module_conuter'))
#        joblib.dump(self.feature, self.module_path+'/'+self.get_option('summary', 'module_feature'))
#        if method in ['SVM', 'NB']:
#            joblib.dump(self.model, self.module_path+'/'+self.get_option('summary', 'module_filename'))
        
        self.pickleop(mode='save')
            
    def pickleop(self, mode='load'):
        if 'load' == mode:
            with open(self.pickle_file, 'rb') as fd:
                data = pickle.load(fd)
                return data
        elif 'save' == mode:
            with open(self.pickle_file, 'wb') as fd:
                pickle.dump(self.tfidf, fd, protocol=2)
    
    def top_n_important(self, quest, top_n=1):
        result_dict = {}
        features = self.tfidf.feature.transform(self.conuter.transform(self.pre_text(quest)))
        word = self.conuter.get_feature_names()#获取词袋模型中的所有词语
        weights = features.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
        #print(word, weights)
        import heapq
        max_ns = heapq.nlargest(top_n, weights)
        for index, weight in enumerate(weights):
            if weight in max_ns:
                result_dict[word[index]] = weight
        return result_dict
    
    def pre_text(self, quests):
        if not isinstance(quests, list):
            quests = [quests]
        return_list = []
        for quest in quests:        
            string = ''
            for word in self.data.word_parser.cut(quest):
                string += (word + ' ')
            return_list.append(string.strip())
        return np.array(return_list)
    
    def predict(self, quest):
        quest = self.pre_text(quest)
        method = self.get_option('summary', 'classfication', 'str')
        if method in ['SVM', 'NB']:
            return self.model.predict(self.feature.transform(self.conuter.transform(quest)))
        elif method in ['similarity']:
            return self.tfidf.predict(quest)
    
    def load_model(self):
        self.tfidf = self.pickleop(mode='load')
        self.conuter = self.tfidf.vectorizer
        self.feature = self.tfidf.feature
        method = self.get_option('summary', 'classfication', 'str')
        if method in ['SVM', 'NB']:
            self.model = self.tfidf.model
#        self.conuter = joblib.load(self.module_path+'/'+self.get_option('summary', 'module_conuter'))
#        self.feature = joblib.load(self.module_path+'/'+self.get_option('summary', 'module_feature'))
#        method = self.get_option('summary', 'classfication', 'str')
#        if method in ['SVM', 'NB']:
#            self.model = joblib.load(self.module_path+'/'+self.get_option('summary', 'module_filename'))
#        elif method in ['similarity']:
#            #
    
    def get_sets_accracy(self, testfile='NONAME'):
        train_total_quest = 0
        train_correct_count = 0
        test_total_quest = 0
        test_correct_count = 0
        if 2 == self.args.get_data_method:
            if 'NONAME' == testfile:
                testfile = self.data.file_path['data_2']
            with open(testfile, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip().split('\t')
                    if line[0].isdigit():
                        label_id = int(line[0])
                    else:
                        self.print_log('format error is {}'.format(line[0]))
                        label_id = 0
                
                    for quest in line[2].split('#&#&#'):
                        train_total_quest += 1
                        predict = self.predict(quest)[0]
                        if label_id == predict:
                            train_correct_count += 1
                    
                    for quest in line[3].split('#&#&#'):
                        test_total_quest += 1
                        predict = self.predict(quest)[0]
                        if label_id == predict:
                            test_correct_count += 1
        
        self.accuracy = (train_correct_count + test_correct_count) / (train_total_quest + test_total_quest)
        self.print_log('{}, {}, {}, {}'.format(train_correct_count, test_correct_count, train_total_quest, test_total_quest))
        self.print_log('最终的准确率为:{},{},{}'.format(test_correct_count / test_total_quest, train_correct_count / train_total_quest, self.accuracy))
    
    def get_accuracy_rate(self):
        pass
#        result = []
#        with open(self.data.file_path['final_test_file'], 'r', encoding='utf-8') as f:
#            for line in f.readlines():
#                line = line.strip().split('\t')
#                predict = self.predict(line[0])
#                result.append(predict[0][predict])#predict是预测的概率最大分类的下标,所以此处是probs[0][predict]最大的概率
#        #允许90%的闲聊数据通过
#        acc_1 = list(sorted(result)[int(len(result)*0.9)])[0]#result里是array格式,将概率设置按照从小到大排序,得到下标为90%的概率,这儿的list(x)[0]是干啥??
#        
#        if hasattr(self.data, 'min_accuracy'):
#            acc_2 = self.data.min_accuracy
#        elif hasattr(self, 'min_accuracy'):
#            acc_2 = self.min_accuracy
#        else:
#            acc_2 = 0.00001
#        
#        if acc_2 > acc_1:
#            self.accuracy = (acc_1 + acc_2)/2
#        else:
#            self.accuracy = acc_2
#        self.print_log('最终的准确率阈值为:{},训练数据最低准确率要求:{},过滤掉90%的闲聊数据的准确率要求:{}'.format(self.accuracy, acc_2, acc_1))