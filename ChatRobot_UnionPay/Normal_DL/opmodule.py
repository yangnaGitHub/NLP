# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 18:28:43 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import tensorflow as tf
from Normal_DL.module import NormalDL as NormalDL
from datetime import timedelta
import os
import traceback
import time
import numpy as np
import configparser

class DLOp():
    def __init__(self, args, data, log=None, **result_dict):
        self.module_name = 'Normal_DL'
        self.model = None
        self.session = None
        self.args = args
        self.data = data
        self.log = log
        self.autodict = result_dict
        self.params = configparser.ConfigParser()
        if self.args.isoptionexist('use_model', 'module_conf'):
            module_conf = self.args.get_option('use_model', 'module_conf', 'str')
        else:
            module_conf = 'module.cfg'
        self.params.read('./' + self.module_name + '/' + module_conf)
        #self.params.read('module.cfg')
        self.module_path = self.get_option('summary', 'module_path', 'str')
        self.need_check = ['input_size', 'result_module']
        for need in self.need_check:
            if need not in result_dict:
                self.print_log('lack of {}'.format(need))            
    
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
    
    def print_shape(self, var, feed_dict=None, prefix='shape'):
        self.print_log('{}==>{}'.format(prefix, self.session.run(self.model.get_shape(var), feed_dict=feed_dict)))
    
    def feed_data(self, x_batch, y_batch, keep_prob):
        feed_dict = {
                self.model.input_x: x_batch,
                self.model.input_y: y_batch,
                self.model.keep_prob: keep_prob
                }
        return feed_dict
    
    def evaluate(self, sess, x_, y_, batch_size=128):
        """评估在某一数据上的准确率和损失"""
        #feed_dict = self.feed_data(x_, y_, 1.0)
        #loss, acc, scores, correct_predictions = sess.run([self.model.loss, self.model.acc, self.model.scores, self.model.correct_predictions], feed_dict=feed_dict)
        #return loss, acc, scores, correct_predictions
        loss = 0.0
        scores = []
        correct_predictions = []
        num_batches_per_epoch = int((len(x_)-1)/batch_size) + 1
        acc = np.array([0.0]*num_batches_per_epoch)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(x_))#最后一组
            batch_x = x_[start_index:end_index]
            batch_y = y_[start_index:end_index]
            feed_dict = self.feed_data(batch_x, batch_y, 1.0)
            tmploss, tmpacc, tmpscores, tmpcorrect_predictions = sess.run([self.model.loss, self.model.acc, self.model.scores, self.model.correct_predictions], feed_dict=feed_dict)
            loss += tmploss
            acc[batch_num] = tmpacc
            scores.extend(tmpscores)
            correct_predictions.extend(tmpcorrect_predictions)
        return loss, np.mean(acc), np.array(scores), np.array(correct_predictions)
        
#        scores_list = []
#        for module in self.DML_methods.values():
#            scores_list.append(module.predict_all(self.args.get_option('use_model', 'result_vaild', 'int'), self.args.get_option('use_model', 'result_batch_size', 'int')))
#        for scores in zip(*scores_list):
#            _, labels = score[0]
#            quests = []
#            for score in scores:
#                tmp, _ = score
#                quests.extend(tmp) 
#            yield quests, labels

    def pre_predict(self, quest, result_method=1, need_hot_vector=0):
        quest_module = []
        try:
            for module in self.autodict['result_module'].values():
                #self.print_log('quest:{}'.format(quest))
                predictions, scores = module.predict(quest, need_hot_vector=need_hot_vector)
                #self.print_log('quest:{},{},{},{}'.format(predictions, predictions.shape, scores, scores.shape))
                if 0 == result_method:
                    quest_module.extend(list(predictions))
                elif 1 == result_method:
                    for score in scores:
                        quest_module.extend(score)
                    #quest_module.extend(list(scores))
        except Exception as e:
            self.print_log(e)
        #self.print_log('module:{}'.format(quest_module))
        return quest_module
        
    def train(self):
        with tf.Graph().as_default() as g:
            self.model = NormalDL(self.args, self.params, self.log, **self.autodict)
            self.session = tf.Session(graph=g)
            with self.session.as_default():
                self.session.run(tf.global_variables_initializer())#初始化所有Variable定义的变量
                saver = tf.train.Saver()#保存模型
                if not os.path.exists(self.module_path):
                    os.makedirs(self.module_path)
                start_time = time.time()
                total_batch = 0              # 总批次
                best_acc_val = 0.0           # 最佳验证集准确率
                last_improved = 0            # 记录上一次提升批次
                require_improvement = int(len(self.data.quest_label.keys()) / self.get_option('summary', 'batch_size', 'int'))   # 如果超过1000轮未提升，提前结束训练
                #require_improvement = self.args.get_option('summary', 'require_improvement', 'int')
                self.print_log('require_improvement: {}'.format(require_improvement))
                
                batches = self.data.get_batch_data(self.get_option('summary', 'test_ratio', 'float'), self.get_option('summary', 'batch_size', 'int'), self.get_option('summary', 'num_epochs', 'int')) 
                
                print_per_batch = int(self.get_option('summary', 'print_per_batch', 'int'))
                for index, batch in enumerate(batches):
                    x_batch, y_batch, x_val, y_val = batch#x_batch, y_batch ==> train                    
                    x_batch = np.array([self.pre_predict(quest) for quest in x_batch])
                    feed_dict = self.feed_data(x_batch, y_batch, self.get_option('summary', 'dropout_keep_prob', 'float'))#dropout_keep_prob正则系数                    
                    #test shape
#                    self.print_shape(self.model.input_x, feed_dict)
#                    self.print_shape(self.model.embedding, feed_dict)
#                    self.print_shape(self.model.outputs[0], feed_dict)
#                    self.print_shape(self.model.outputs[1], feed_dict)
#                    self.print_shape(self.model.alpha, feed_dict)
                    
                    if len(self.data.quest_label) < 1000:#少于1000个问题(原问题和扩展问题算一个)
                        print_per_batch = 1#默认值是50
                    if total_batch % print_per_batch == 0:#以及第一次,多少个batch打印一次
                        #每多少轮次输出在训练集和验证集上的性能
                        feed_dict[self.model.keep_prob] = 1.0
                        loss_train, acc_train = self.session.run([self.model.loss, self.model.acc], feed_dict=feed_dict)
                        loss_val, acc_val, scores, correct_predictions = self.evaluate(self.session, np.array([self.pre_predict(quest) for quest in x_val]), y_val, self.get_option('summary', 'batch_size', 'int'))#x_val,y_val是测试集合
                        
                        if acc_val > best_acc_val:#求最大的准确率
                               # 保存最好结果
                            best_acc_val = acc_val
                            last_improved = total_batch#提升的批次
                            saver.save(sess=self.session, save_path=self.module_path)
                            self.data.get_accuracy_rate(correct_predictions, scores)#计算准确率阈值
        
                        time_dif = self.get_time_dif(start_time)
                        
                        self.print_log('total_batch: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%}, Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5}'.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif))
                     
                    self.session.run(self.model.optimizer, feed_dict = feed_dict)#运行优化
                    total_batch += 1
                    
                    if total_batch == (int(len(self.data.quest_label.keys()) / self.get_option('summary', 'batch_size', 'int')) + 1):
                        last_improved = total_batch
                    if total_batch - last_improved > require_improvement:
                        # 验证集正确率长期不提升，提前结束训练
                        break#跳出循环
                self.print_log('最佳准确率:{}'.format(best_acc_val))
    
    def load_model(self):
        with tf.Graph().as_default() as g:
            self.model = NormalDL(self.args, self.params, self.log, **self.autodict)
            self.session = tf.Session(graph=g)
            with self.session.as_default():
                self.session.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                saver.restore(sess=self.session, save_path=self.module_path)

    def predict(self, quest):
        feed_dict = {self.model.input_x:np.array([self.pre_predict(quest, need_hot_vector=1)]), self.model.keep_prob: 1.0}
        return self.session.run([self.model.predictions, self.model.scores], feed_dict=feed_dict)
    
    def get_sets_accracy(self, testfile='NONAME'):
        train_total_quest = 0
        train_correct_count = 0
        test_total_quest = 0
        test_correct_count = 0
        if 2 == self.args.get_data_method:
            if 'NONAME' == testfile:
                testfile = self.data.file_path['data_2']
            #self.print_log('testfile {}'.format(testfile))
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
                        predict, probs = self.predict(quest)
                        
                        if label_id == self.data.id_to_label[str(predict[0])]:
                            train_correct_count += 1
                    
                    for quest in line[3].split('#&#&#'):
                        test_total_quest += 1
                        predict, probs = self.predict(quest)
                        
                        if label_id == self.data.id_to_label[str(predict[0])]:
                            test_correct_count += 1
        
        self.accuracy = (train_correct_count + test_correct_count) / (train_total_quest + test_total_quest)
        self.print_log('最终的准确率为:{},{},{}'.format(test_correct_count / test_total_quest, train_correct_count / train_total_quest, self.accuracy))
    
    def get_accuracy_rate(self):
        result = []
        with open(self.data.file_path['final_test_file'], 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                feed_dict = {self.model.input_x: np.array([self.pre_predict(line[0], need_hot_vector=1)]), self.model.keep_prob: 1.0}
                predict, probs = self.session.run([self.model.predictions, self.model.scores], feed_dict=feed_dict)
                result.append(probs[0][predict])#predict是预测的概率最大分类的下标,所以此处是probs[0][predict]最大的概率
        #允许90%的闲聊数据通过
        acc_1 = list(sorted(result)[int(len(result)*0.9)])[0]#result里是array格式,将概率设置按照从小到大排序,得到下标为90%的概率,这儿的list(x)[0]是干啥??
        acc_2 = self.data.min_accuracy
#        if hasattr(self.data, 'min_accuracy'):
#            acc_2 = self.data.min_accuracy
#        elif hasattr(self, 'min_accuracy'):
#            acc_2 = self.min_accuracy
#        else:
#            acc_2 = 0.00001
        
        if acc_2 > acc_1:
            self.accuracy = (acc_1 + acc_2)/2
        else:
            self.accuracy = acc_2
        self.print_log('最终的准确率阈值为:{},训练数据最低准确率要求:{},过滤掉90%的闲聊数据的准确率要求:{}'.format(self.accuracy, acc_2, acc_1))