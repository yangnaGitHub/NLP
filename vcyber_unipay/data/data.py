# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 17:36:30 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import pymysql
import pandas as pd
from collections import defaultdict
from collections import Counter
from data.wordparser import word_parser
import tensorflow.contrib.keras as kr
import numpy as np
import time
import os
import traceback
import pickle

class data():
    def __init__(self, args):
        self.args = args
        
        self.print_log('data initial')
        
        self.get_file_path()
        
        self.data_process()
    
    def print_log(self, message):
        if self.args.local_debug:
            print('[%s][%s:%s]' % (os.path.basename(__file__), self.__class__.__name__, traceback.extract_stack()[-2][2]), message)
        if self.args.print_to_log:
            pass
    
    def get_file_path(self):
        section = 'data_dir'
        data_folder_str = 'data_folder'
        self.file_path = {}
        if self.args.issectionexist(section):
            alloptions = self.args.getAlloption(section)
            if self.args.isoptionexist(section, data_folder_str):
                data_folder = self.args.get_option(section, data_folder_str, 'str')
            else:
                data_folder = './data/'
            for option in alloptions:
                if data_folder_str == option[0]:
                    continue
                self.file_path[option[0]] = data_folder + self.args.get_option(section, option[0], 'str')

    def data_process(self):
        '''
        数据处理函数,建立字典等初始化动作
        '''
        self.data_process_start()
        
        self.get_data_pre()
        self.get_data()

        self.build_vocab_size()
        self.get_labels()

        self.data_process_end()
        
    def data_process_start(self):
        '''
        数据处理之前要准备的
        '''
        #构建好统一词表
        self.word_parser = word_parser(self)
        if 1 == self.args.use_same_words:
            self.same_words = defaultdict(set)
            file_names = self.get_samewords_filename()
            for name in file_names:
                with open(name, 'r', encoding='utf-8') as f: 
                    for line in f.readlines():
                        line = line.upper().strip().split('=')
                        if len(line) <= 1:
                            continue
                        words = line[1].strip().split(";")
                        if len(words) <= 1:
                            continue
                        self.same_words[line[0]].update(words)#是列表所以要用update,和前面add方法的区别
    
    def pickleop(self, mode='load'):
        if 'load' == mode:
            with open(self.pickle_file, 'rb') as fd:
                data = pickle.load(fd)
                return data
        elif 'save' == mode:
            if 'sort' in self.pickle_variable.keys():
                variable_sort = self.pickle_variable['sort']
            else:
                return
            savelist = [self.pickle_variable]
            for variable in variable_sort:
                if variable in self.pickle_variable.keys():
                    options = self.pickle_variable[variable]
                    if variable in ['data']:
                        savelist.extend([getattr(self, option) for option in options if hasattr(self, option)])
                        #savelist.extend([eval('self.' + option) for option in options])
                    else:
                        rootobject = getattr(self, variable) if hasattr(self, variable) else None
                        savelist.extend([getattr(rootobject, option) for option in options if hasattr(rootobject, option)])
                        #savelist.extend([eval('self.' + variable + '.' + option) for option in options])
            savetuple = tuple(savelist)
            with open(self.pickle_file, 'wb') as fd:
                pickle.dump(savetuple, fd, protocol=2)
    
    def pickle_object(self, pickle_module='data_pickle'):
        if self.args.isoptionexist(pickle_module, 'pickle_path'):
            self.pickle_path = self.args.get_option(pickle_module, 'pickle_path', 'str')
        else:
            self.pickle_path = './model/pickle/data'
        
        self.pickle_path += '/'
        if not os.path.exists(self.pickle_path):
            os.makedirs(self.pickle_path)
        
        if self.args.isoptionexist(pickle_module, 'pickle_file'):
            self.pickle_file = self.pickle_path + self.args.get_option(pickle_module, 'pickle_file', 'str')
        else:
            self.pickle_file = self.pickle_path + 'data.pickle'
        
        self.pickle_variable = {}
        Alloptions = self.args.getAlloption(pickle_module)
        variable_sort = []
        for options in Alloptions:
            if options[0].startswith('variable_'):
                objectname = options[0][options[0].find('_')+1:]
                self.pickle_variable[objectname] = [variable.strip() for variable in options[1].split(',')]
                variable_sort.append(objectname)
        self.pickle_variable['sort'] = variable_sort
        self.pickleop(mode='save')
        
    def data_process_end(self):
        '''
        数据处理结束要做的
        '''
        self.pickle_object()
        
        if self.args.local_debug:
            self.debug_print_var()
    
    def debug_print_var(self, print_count=50, mid_split = ': '):
        with open(self.file_path['debug'], 'w', encoding='utf-8') as fd:
            fd.write('same_words length[%d]\n' % len(self.same_words))
            for index, key in enumerate(self.same_words.keys()):
                fd.write(key + mid_split + str(self.same_words[key]) + '\n')
                if index > print_count:
                    break
            
            fd.write('datas_len[%d] vocab_size[%d] max_document_lenth[%d] num_class[%d]\n' % (len(self.datas), self.args.vocab_size, self.args.max_document_lenth, self.args.num_class))
            
            fd.write('quests length[%d]\n' % len(self.datas))
            for index, quest in enumerate(self.quests):
                fd.write(quest + '\n')
                if index > print_count:
                    break
            
            fd.write('word_to_id length[%d]\n' % len(self.word_to_id))
            for index, word in enumerate(self.word_to_id.keys()):
                fd.write(word + mid_split + str(self.word_to_id[word]) + '\n')
                if index > print_count:
                    break
            
            fd.write('label_to_id length[%d]\n' % len(self.label_to_id))
            for index, label in enumerate(self.label_to_id.keys()):
                fd.write(str(label) + mid_split + str(self.label_to_id[label]) + '\n')
                if index > print_count:
                    break
    
    def extend_by_sameword_no(self,quest):
        return [quest]
    
    def add_outdata(self):
        if 0 == self.args.add_outside_sets:
            return
        
        label_id = 0
        for outdata in self.outdatas:
            outdata = outdata.split('\t')
            for index, content in enumerate(outdata):
                for exp_content in self.extend_by_sameword_no(content):
                    self.args.max_document_lenth = max(len(exp_content), self.args.max_document_lenth)
                    if ((index == len(outdata) - 1) and (self.args.get_data_method in [2])):
                        self.testquests.add(exp_content)
                        self.label_testquest[label_id].add(exp_content)
                        self.testquest_label[exp_content] = label_id
                        continue
                    self.quests.add(exp_content)
                    self.label_quest[label_id].add(exp_content)
                    self.quest_label[exp_content] = label_id
        self.print_log('add out count:{}, {}'.format(len(self.quests), len(self.testquests)))
    
    def get_data(self):
        '''
        清洗数据并import到Python中
        '''
        if 2 == self.args.get_data_method:
            min_length = 4
            quest_count = 0
            test_count = 0
            for data in self.datas:
                data = data.split('\t')
                if len(data) < min_length:
                    continue
                
                if data[0].isdigit():
                    label_id = int(data[0])
                else:
                    self.print_log('format error is ('+ str(data) + ')')
                    label_id = 0
                #train
                for content in (data[2]).split('#&#&#'):
                    quest_count += 1
                    for exp_content in self.extend_by_sameword(content):
                        self.quests.add(exp_content)
                        self.label_quest[label_id].add(exp_content)
                        self.quest_label[exp_content] = label_id
                        self.args.max_document_lenth = max(len(exp_content), self.args.max_document_lenth)
                #test
                for content in (data[3]).split('#&#&#'):
                    test_count += 1
                    for exp_content in self.extend_by_sameword(content):
                        self.testquests.add(exp_content)
                        self.label_testquest[label_id].add(exp_content)
                        self.testquest_label[exp_content] = label_id
                        self.args.max_document_lenth = max(len(exp_content), self.args.max_document_lenth)
            self.print_log('expend count:{}, {}, {}, {}'.format(quest_count, len(self.quests), test_count, len(self.testquests)))
            self.add_outdata()
            return
        
        for index in range(len(self.datas)):
            if 1 == self.args.get_data_method:
                label_id = int(self.datas.iloc[index].qa_id)
                #content_id = int(datas.iloc[index].id)
                content = self.datas.iloc[index].content
            elif 0 == self.args.get_data_method:    
                label_id = int(self.datas[index][0])
                #content_id = int(datas[index][1])
                content = self.datas[index][2]
            try:
                content = self.data_clear(content, 'content')
            except Exception as e:
                self.print_log(e)
            for exp_content in self.extend_by_sameword(content):
                self.quests.add(exp_content)
                self.label_quest[label_id].add(exp_content)
                self.quest_label[exp_content] = label_id
                self.args.max_document_lenth = max(len(exp_content), self.args.max_document_lenth)#最长问题
        self.add_outdata()

    def get_data_pre(self):
        '''
        获取数据之前要做的操做,准备数据并准备容器
        '''
        self.label_quest = defaultdict(set)
        self.quest_label = {}
        self.quests = set()
        
        if 1 == self.args.get_data_method:
            self.datas = pd.read_csv(self.file_path['data_1'], sep='\t')
        elif 0 == self.args.get_data_method:
            db = pymysql.connect(host="192.168.1.245",port=3306,user="robosay",password="robosay",db="platform",charset="utf8")
            cursor = db.cursor() 
            cursor.execute('''
                           select kqq.qa_id,kq.id,kq.content
                           from kb_question kq,kb_qa_question kqq
                           where kq.id = kqq.question_id 
                           and kq.is_delete = '0'
                           and kqq.is_delete = '0'
                           and kqq.qa_id in(select id from kb_qa where parent_id in(select id from kb_scml where type='0' and user_id = %d and is_delete = '0'));
                           '''%(self.args.user_id))
            self.datas = cursor.fetchall()
            db.close()
        elif 2 == self.args.get_data_method:
            self.testquests = set()
            self.testquest_label = {}
            self.label_testquest = defaultdict(set)
            with open(self.file_path['data_2'], 'r', encoding='utf-8') as fd:
                self.datas = fd.readlines()
        
        if self.args.add_outside_sets:
            with open(self.file_path['out_data'], 'r', encoding='utf-8') as fd:
                self.outdatas = fd.readlines()
        
    def get_samewords_filename(self):
        file_names = []
        file_names.append(self.file_path['common_same'])
        if self.args.use_same_file_class == 1:
            file_names.append(self.file_path['car_same'])
        elif self.args.use_same_file_class == 2:
            file_names.append(self.file_path['bank_same'])
        return file_names
        
    def data_clear(self, data, description='None'):
        '''
        数据清洗
        '''
        if 'None' == description:
            return None
        elif 'content' == description:
            if data.endswith('.0') or data.endswith('.00'):#增加对数字错误代码的处理 思考:数字代码的处理是不是还要做其他操作呢<==因为出现这种数字代码就基本上要命中这个问题,根据最后的结果来看这边是不是要优化
                data = data[:data.rfind('.')]
            return data.replace(' ','').strip().upper()
    
    def build_vocab_size(self):
#        for quest in self.quests:
#            allquest += quest
#        if self.args.get_data_method in [2]:
#            for quest in self.testquests:
#                allquest += quest
        #self.print_log('quests:{}, testquests:{}'.format(self.quests, self.testquests))
        
        if 1 == self.args.use_cut_words:
            count_words = {}
            for quest in (self.quests | self.testquests if self.args.get_data_method in [2] else self.quests):
                for word in quest:
                    if word in count_words:
                        count_words[word] += 1
                    else:
                        count_words[word] = 1
        else:
            allquest = ''
            for quest in (self.quests | self.testquests if self.args.get_data_method in [2] else self.quests):
                allquest += quest
        
        with open(self.file_path['chat'], 'r', encoding='utf-8') as f:#有木有	这个木有	这个没有]	这个没有 
            for line in f.readlines():
                if 1 == self.args.use_cut_words:
                    for word in self.word_parser.cut(line.upper().strip().split('\t')[0]):
                        if word in count_words:
                            count_words[word] += 1
                        else:
                            count_words[word] = 1
                else:
                    allquest += line.upper().strip().split('\t')[0]#有木有 为何只有第一列
        
        if 1 == self.args.use_cut_words:
            words, count = np.array(list(count_words.keys())), np.array(list(count_words.values())).argsort()
            words = [words[count[index]] for index in range(len(count)-1, -1, -1)]
        else:
            if len(allquest) != 0:
                counter = Counter(allquest)
                count_pairs = counter.most_common(self.args.vocab_size - 1)
                words, _ = list(zip(*count_pairs))
            else:
                words = []
        
        words = ['<UNK>'] + list(words)
        self.word_to_id = dict(zip(words, range(len(words))))#字分配递增ID(UNK为0号,最频繁的字分为1号,一直到最后)
        self.args.vocab_size = len(words)
        #self.print_log('vocab_size:{}, word_to_id:{}'.format(self.args.vocab_size, self.word_to_id))
    
    def get_labels(self):
        self.label_to_id = dict(zip(self.label_quest.keys(), range(len(self.label_quest))))#给label分配递增ID
        self.id_to_label = {str(v):k for k,v in self.label_to_id.items()}#为何要转换成str的ID??
        self.args.num_class = len(self.label_quest)#有多少类
        #self.print_log('num_class:{}, id_to_label:{}'.format(self.args.num_class, self.id_to_label))
    
    def extend_by_sameword(self,quest):
        if 0 == self.args.use_same_words:#init的时候的同义词扩展
            if 1 == self.args.use_cut_words:
                string = ''
                for word in self.word_parser.cut(quest):
                    string += (word + ' ')
                return [string.strip()]
            else:
                return [quest]
        local_sentences = []
        #查找同义词词表
        for word in self.word_parser.cut(quest):#按照word_parser分词 ==> 如何 开通 银联 在线 支付
            temp = word
            for _,item in self.same_words.items():#item是一个集合(同义词的集合)(如何,怎么,怎样)
                if word in item:#每个词一个一个比较会不会比较慢,可不可以想办法提高一下
                    temp = list(item)#[如何,怎么,怎样]
           #[['怎么样', '怎样', '如何', '咋样', '何如'], '开', '通', '在', '线', '支', '付'] len=7
            local_sentences.append(temp)#append会保持数据原有的形状,注意个expend的区别
        #扩充语句
        quests = self.extend_quests(quest, local_sentences)
        return quests
    
    def extend_quests(self, quest, local_sentences):#比如(如何开通银联在线支付, [[如何,怎么,怎样], [支付, 付钱]])
        quests = set()#set([quest])
        if 0 == len(local_sentences):
            return quests
        #遍历
        self.get_next_word(0, '', quests, local_sentences)
        return quests
    
    def get_next_word(self, index, quest, quests, local_sentences):#('如何开通银联在线支付', '如何付钱', '如何支付', '怎么付钱', '怎么支付', '怎样付钱', '怎样支付')
        if index == len(local_sentences):
            quests.add(quest.strip())
            return              
        word = local_sentences[index] #[如何,怎么,怎样]
        if isinstance(word, list):
            for w in word:
                if 1 == self.args.use_cut_words:
                    self.get_next_word(index+1, quest+' '+w, quests, local_sentences)#列表中的属于一个近义词
                else:
                    self.get_next_word(index+1, quest+w, quests, local_sentences)#列表中的属于一个近义词
        else:
            if 1 == self.args.use_cut_words:
                self.get_next_word(index+1, quest+word+' ', quests, local_sentences)#其他不适近义词的
            else:
                self.get_next_word(index+1, quest+word, quests, local_sentences)#其他不适近义词的
    
#    def get_all_quests_and_label(self, num_epochs, batch_size):
#        if 2 == self.args.get_batch_method:
#            quests, labels = (list(self.quest_label.keys()) + list(self.testquest_label.keys())), (list(self.quest_label.values()) + list(self.testquest_label.values()))
#        else:
#            quests, labels = list(self.quest_label.keys()), list(self.quest_label.values())
#        num_batches_per_epoch = int((len(quests)-1)/batch_size) + 1
#        for epoch in range(num_epochs):
#            for batch_num in range(num_batches_per_epoch):
#                start_index = batch_num * batch_size
#                end_index = min((batch_num + 1) * batch_size, len(quests))#最后一组
#                batch_x = quests[start_index:end_index]
#                batch_y = labels[start_index:end_index]
#                batch_x,batch_y = self.build_vector(batch_x,batch_y)
#                yield batch_x, batch_y
    
    def range_index(self, idx, size, window_size=5):
        target_window = np.random.randint(1, window_size+1)
        start_point = idx - target_window if (idx - target_window) > 0 else 0#上限溢界处理
        end_point = idx + target_window if (idx - target_window) < size else size
        return start_point, end_point
    
    #embedding按照字+要添加(词)    
    def get_embed_batch_data(self, test_ratio, batch_size, num_epochs, startpos=5000, window_size=3):
        quests = np.array(list(self.quest_label.keys()))
        if 0 == self.args.get_batch_method:
            pass
        elif 1 == self.args.get_batch_method:
            pass
        elif 2 == self.args.get_batch_method:
            testquests = np.array(list(self.testquest_label.keys()))
            shuffle_indices = np.random.permutation(np.arange(len(self.quest_label)))#下标
            shuffle_indices_test = np.random.permutation(np.arange(len(self.testquest_label)))
            train_x = quests[shuffle_indices]#train
            #train_x, train_y = quests, labels
            test_x = testquests[shuffle_indices_test]
            #test_x, test_y = testquests, testlabels
            #test_x, test_y = self.build_vector(test_x, test_y)
        elif 3 == self.args.get_batch_method:
            pass
        
        for epoch in range(num_epochs):
            for quest in (set(train_x) | set(test_x)):#是否要添加额外的
                batch_x, batch_y = [], []
                quest = self.build_one_vector(quest)
                for index, x in enumerate(quest):
                    start_point, end_point = self.range_index(index, len(quest), window_size)
                    temp_y = list(set(list(quest[start_point: index]) + list(quest[index+1: end_point+1])))
                    batch_y.extend(temp_y)
                    batch_x.extend([x]*len(temp_y))
                yield batch_x, batch_y
    
    def get_batch_data(self, test_ratio, batch_size, num_epochs, startpos=5000):
        """生成批次数据"""
        quests, labels = np.array(list(self.quest_label.keys())), np.array(list(self.quest_label.values()))
        startpos = len(self.quest_label) - int(len(self.quest_label) * test_ratio)
        if 0 == self.args.get_batch_method:
            #函数shuffle与permutation都是对原来的数组进行重新洗牌
            #区别在于shuffle直接在原来的数组上进行操作,改变原来数组的顺序,无返回值.而permutation不直接在原来的数组上进行操作,而是返回一个新的打乱顺序的数组,并不改变原来的数组
            shuffle_indices = np.random.permutation(np.arange(len(self.quest_label)))#下标
            #生成训练数据                                                
            train_x, train_y = quests[shuffle_indices], labels[shuffle_indices]#train
            #再次随机取测试集合
            shuffle_indices = np.random.permutation(np.arange(len(self.quest_label)))[0:min(len(self.quest_label), startpos)]#不够5000就有多少取多少,多于5000就最多取5000
            test_x, test_y = quests[shuffle_indices], labels[shuffle_indices]
            test_x, test_y = self.build_vector(test_x, test_y)
            self.print_log('shuffle_indices:{},train_x:{},train_y:{},test_x:{},test_y:{}'.format(shuffle_indices.shape, train_x.shape, train_y.shape, test_x.shape, test_y.shape))
        elif 1 == self.args.get_batch_method:
            shuffle_indices = np.random.permutation(np.arange(len(self.quest_label)))                                           
            train_x, train_y = quests[shuffle_indices[:startpos]], labels[shuffle_indices[:startpos]]
            test_x, test_y = quests[shuffle_indices[startpos:]], labels[shuffle_indices[startpos:]]
            test_x, test_y = self.build_vector(test_x, test_y)
            self.print_log('startpos:{},train_x:{},train_y:{},test_x:{},test_y:{}'.format(startpos, train_x.shape, train_y.shape, test_x.shape, test_y.shape))
        elif 2 == self.args.get_batch_method:
            testquests, testlabels = np.array(list(self.testquest_label.keys())), np.array(list(self.testquest_label.values()))
            #打乱一下顺序
            shuffle_indices = np.random.permutation(np.arange(len(self.quest_label)))#下标
            shuffle_indices_test = np.random.permutation(np.arange(len(self.testquest_label)))
            
            #是否打乱顺序
            train_x, train_y = quests[shuffle_indices], labels[shuffle_indices]#train
            #train_x, train_y = quests, labels
            test_x, test_y = testquests[shuffle_indices_test], testlabels[shuffle_indices_test]
            #test_x, test_y = testquests, testlabels
            
            test_x, test_y = self.build_vector(test_x, test_y)
            self.print_log('train_x:{},train_y:{},test_x:{},test_y:{}'.format(train_x.shape, train_y.shape, test_x.shape, test_y.shape))
        elif 3 == self.args.get_batch_method:
            train_x, train_y = quests[startpos:], labels[startpos:]
            test_x, test_y = quests[:startpos], labels[:startpos]
            test_x, test_y = self.build_vector(test_x, test_y)
            self.print_log('train_x:{},train_y:{},test_x:{},test_y:{}'.format(train_x.shape, train_y.shape, test_x.shape, test_y.shape))
        
        num_batches_per_epoch = int((len(train_x)-1)/batch_size) + 1
        if 4 == self.args.get_batch_method:
            num_batches_per_epoch = int(startpos-1/batch_size) + 1
        self.print_log('num_batches_per_epoch:{}'.format(num_batches_per_epoch))
        
        for epoch in range(num_epochs):#所有的batch训练多少圈
            #将所有的数据打乱顺序,重新来选择
            if 4 == self.args.get_batch_method:
                quests_shuffle_indices = np.random.permutation(quests)
                labels_shuffle_indices = np.random.permutation(labels)
                train_x, train_y = quests_shuffle_indices[startpos:], labels_shuffle_indices[startpos:]
                test_x, test_y = quests_shuffle_indices[:startpos], labels_shuffle_indices[:startpos]
                test_x, test_y = self.build_vector(test_x, test_y)
                            
            self.print_log('Epoch:{}'.format(epoch + 1))
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, len(train_x))#最后一组
                batch_x = train_x[start_index:end_index]
                batch_y = train_y[start_index:end_index]
                batch_x,batch_y = self.build_vector(batch_x,batch_y)
                one_time = time.time()
                yield batch_x, batch_y, test_x, test_y#每次产生一组训练数据
                one_time = time.time()-one_time
                #这个时间为啥这么计算
                #正确时间+1
                self.args.time = round((one_time*((num_epochs-epoch)*(num_batches_per_epoch)-batch_num))/60) + 1
            self.args.train_rate = (epoch)/num_epochs
    
    def build_vector(self, data, label):#问题和label
        """将文件转换为id表示"""
        data_id, label_id = [], []
        for i in range(len(data)):
            if label[i] not in self.label_to_id:#label和ID的对应关系(如果这个label不存在)
                continue
            vector = []
            for x in (data[i].split(' ') if 1 == self.args.use_cut_words else data[i]):#data[i]每个问题,所以x是每个字
                if x in ['\n', '\t', '']:
                    continue
                vector.append(self.word_to_id.get(x, self.word_to_id['<UNK>']))#build_vocab_size,字分配每个递增ID,虽大4999(5000个),如果这个字没找到就返回UNK的ID 0,[200, 1, 5, 1000, ...]根据出现的频次来编序列的vector
            data_id.append(vector)#一个问题按照字转化成数字<==文字要变成可处理的数据必须要做处理(计算机只认识数字)==>每句话的长度不一致,所以data_id不一致
            label_id.append(self.label_to_id[label[i]])#label也转化成分配的ID,便于处理和计算
        #使用keras提供的pad_sequences来将文本pad为固定长度
        #预处理(填充序列pad_sequences)
        #max_document_lenth为序列的最大长度,大于此长度的序列将被截短,小于此长度的序列将在后部填value(self.word_to_id['<UNK>'])
        #truncating:‘pre’或‘post’,确定当需要截断序列时,从起始还是结尾截断
        #padding:‘pre’或‘post’,确定当需要补0时,在序列的起始还是结尾补,默认是在前面补
        x_pad = kr.preprocessing.sequence.pad_sequences(data_id, self.args.max_document_lenth, value=self.word_to_id['<UNK>'])#若词典中的词超过5000个,没有出现的词都是UNK,这边不够的也处理成UNK是不是不够合理
        #一系列有用工具
        #类别向量(从0到num_classes(len(self.label_to_id))的整数向量)映射为二值类别矩阵(就是0-1表示的,就是one-hot编码)
        y_pad = kr.utils.to_categorical(label_id, num_classes=len(self.label_to_id))#将标签转换为one-hot表示
        return x_pad, y_pad
    
    def build_one_vector(self, raw_quest):
        raw_quest = raw_quest.upper()
        #重载的时候要序列化
        quest = [self.word_to_id.get(word, self.word_to_id['<UNK>']) for word in (self.word_parser.cut(raw_quest).split(' ') if 1 == self.args.use_cut_words else raw_quest) if word not in ['\n', '\t', '']]
        if len(quest) >= self.args.max_document_lenth:
            quest = quest[:self.args.max_document_lenth]#截取
        else:
            #pad_sequences补0是往前面补
            quest = (self.args.max_document_lenth-len(quest))*[self.word_to_id['<UNK>']] + quest#前面是UNK
        #self.print_log('问题:{}\n向量:{}'.format(raw_quest,quest))
        return np.array(quest)
    
    def get_accuracy_rate(self, correct_predictions, scores):#(batch_size), (batch_size, num_class)
        correct_predictions = list(correct_predictions.astype(np.int32))
        indexs = np.array([index for index,i in enumerate(correct_predictions) if i == 1])#判断正确的问题的索引
        #self.print_log('scores[indexs]:{}'.format(scores[indexs]))
        temp = [max(s) for s in scores[indexs]]#每个问题最大的得分[batch_size]
        #self.print_log('temp:', scores[indexs], temp)
        self.min_accuracy = min(temp)
        self.print_log('准确率最低要求:{},平均准确率:{},最高准确率:{}'.format(min(temp),np.mean(temp),max(temp)))