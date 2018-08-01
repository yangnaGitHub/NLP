# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:50:16 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

from data.data import data as data
from TextCNN.opmodule import TextCNNOp
from LSTM.opmodule import LSTMOp
from TFIDF.opmodule import TFIDFOp
from Result.opmodule import ResultOp
from log import *
from threading import Thread
import os
import traceback
import pickle

class module():
    def __init__(self, args, initdata):
        self.args = args
        self.data = initdata
        self.log = log(args)
        self.DML_methods = {}
        self.make_mothod()
    
    def print_log(self, message):
        if self.args.local_debug:
            print('[%s][%s:%s]' % (os.path.basename(__file__), self.__class__.__name__, traceback.extract_stack()[-2][2]), message)
        if self.args.print_to_log:
            if self.log:
                self.log.print_to_file(message)
    
    def make_mothod(self):
        section = 'use_model'
        model_name_str = 'model_name'
        result_vaild_str = 'result_vaild'
        self.DML_methods_str = []
        self.DML_methods_Vaild = []
        if self.args.isoptionexist(section, model_name_str):
            model_names = self.args.get_option(section, model_name_str, 'str')
            for name in model_names.split(','):
                self.DML_methods_str.append(name.strip())
        else:
            self.print_log('model_name not exist')
        if self.args.isoptionexist(section, result_vaild_str):
            result_vaild = self.args.get_option(section, result_vaild_str, 'str')
            for name in result_vaild.split(','):
                self.DML_methods_Vaild.append(name.strip())
    
    def Deal_Result(self):
        pass
        #return self.DML_methods_result['TextCNNOp' + '_predict'], self.DML_methods_result['TextCNNOp' + '_predict']
    
    def restore_object(self, pickle_module='data_pickle'):
        if self.args.isoptionexist(pickle_module, 'pickle_path'):
            self.pickle_path = self.args.get_option(pickle_module, 'pickle_path', 'str')
        else:
            self.pickle_path = './model/pickle/data'
        if self.args.isoptionexist(pickle_module, 'pickle_file'):
            self.pickle_file = self.pickle_path + self.args.get_option(pickle_module, 'pickle_file', 'str')
        else:
            self.pickle_file = self.pickle_path + 'data.pickle'
        
        with open(self.pickle_file, 'rb') as fd:
            data = pickle.load(fd)
        
        if 'sort' in self.data[0].keys():
            variable_sort = self.data[0]['sort']
        else:
            return
        
        index = 1
        for variable in variable_sort:
            if variable in data[0].keys():
                opitons = data[0][variable]
                for option in opitons:
                    eval('self.' + variable + '.' + option) = data[index]
                    index += 1
    
    def predict(self, **kwargs):
        if 0 == self.args.local_method:
            quest = kwargs['params']['quest']
        elif 1 == self.args.local_method:
            if self.args.isoptionexist('local_params', 'chat_quest'):
                quest = self.args.get_option('local_params', 'chat_quest', 'str')
            else:
                self.print_log('local test and question can`t find')
        
        params = {}
        self.DML_methods_result = {}
        for method in self.DML_methods_str:
#            if self.DML_methods[method].get_model() is None:
#                params['answer']='深度学习没有导入数据，无法提供对话功能'
#                params['success']="false"
#                return params
            
            #Notice:不用再次训练,用已经序列话好的参数和模型直接预测,这边还原参数要不要放到各个模型(可能每个模型保存的参数不一致)
            if (not self.DML_methods) or (self.DML_methods[method].get_model() is None):
                self.print_log('this model not avliable, it will load exist model')
                if self.data is None:
                    self.print_log('data are not avliable')
                    self.restore_object()
                self.DML_methods[method] = eval(method + '(self.args, self.data, self.log)')
                self.DML_methods[method].load_model(savedata)

            try:
                self.DML_methods_result[method + '_predict'], self.DML_methods_result[method + '_probs'] = self.DML_methods[method].predict(quest)
                predict, probs = self.DML_methods_result[method + '_predict'], self.DML_methods_result[method + '_probs']#self.Deal_Result()
                if 0 == self.args.local_method:
                    if kwargs['params']['rate'] !=0:
                        if probs[0][predict] >= kwargs['params']['rate']:
                            params['success'] = "true"
                            params['answer'] = self.data.id_to_label[str(predict[0])]
                        else:
                            params['answer'] = '准确率低于要求，结果不可用'
                            params['success'] = "false"
                    else:
                        if probs[0][predict] >= self.accuracy:
                            params['success'] = "true"
                            params['answer'] = self.data.id_to_label[str(predict[0])]
                        else:
                            params['answer'] = '准确率低于要求，结果不可用'
                            params['success'] = "false"
                elif 1 == self.args.local_method:
                    self.print_log('predict:{}, {}'.format(str(predict[0]), self.data.id_to_label[str(predict[0])]))
            except Exception as e:
                self.print_log(e)
                params['answer'] = '使用对话功能前请先训练模型'
                params['success'] = "false"
                return params
    
    def train(self, **kwargs):
        if self.data is None:#这儿应该判断数据是否更新来判断是不是要初始化
            self.data = data(self.args)
        
        params={}
        
        if 0 == len(self.data.quest_label):
            params['success'] = "false"
            params['message'] = "深度学习没有数据，不能训练"
        else:
            #启动一个线程开始重新训练
            if 0 == self.args.local_method:
                t = Thread(target = self.retrain)
                t.start()
            elif 1 == self.args.local_method:
                self.retrain()

            params['success'] = "true"
            params['message'] = "retrain start"
        return params
    
    def retrain(self):
        self.print_log('retrain start')
        try:
            if self.DML_methods:
                self.DML_methods.clear()
            self.print_log('DML_methods_str{}'.format(self.DML_methods_str))
            for method in self.DML_methods_str:
                self.DML_methods[method] = eval(method + '(self.args, self.data, self.log)')
                self.DML_methods[method].train()
                self.DML_methods[method].load_model()
                self.DML_methods[method].get_sets_accracy()
                self.DML_methods[method].get_accuracy_rate()

            self.args.rate=1.0
            self.args.time=0
        except Exception as e:
            self.args.rate=1.0
            self.args.time=0
            self.print_log(e)

    def lookup(self, **kwargs):
        params = {} 
        params['progress'] = ("%d%%")%(self.args.rate*100)
        #以分钟计算
        params['need_time'] = self.args.time
        params['success'] = 'true'
        return params
        
    def start_log(self, **kwargs):
        if kwargs['params']['log'] == 'true':
            self.args.print_to_log = 1
            self.print_log('现在开启log日志记录功能')
        else:
            self.args.print_to_log = 0
            self.print_log('现在关闭log日志记录功能')
        params={}
        params['success']='true'
        return params