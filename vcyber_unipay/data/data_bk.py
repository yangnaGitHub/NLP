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
    #用作测试序列化是否OK
    def pickleop(self, mode='load'):
        if 'load' == mode:
            with open(self.pickle_file, 'rb') as fd:
                data = pickle.load(fd)
                return data
        elif 'save' == mode:
            #self.data.id_to_label
            #self.data.word_to_id
            #self.data.min_accuracy
            #self.args.max_document_lenth = 122
            #self.args.num_class = 561
            #self.args.vocab_size = 4012
            savetuple = (self.id_to_label, self.word_to_id, self.min_accuracy, self.args.max_document_lenth, self.args.num_class, self.args.vocab_size)
            with open(self.pickle_file, 'wb') as fd:
                pickle.dump(savetuple, fd, protocol=2)

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
        
    def test_pickle(self):
        return
        #用作测试序列化是否OK
        self.pickle_file = 'natasha.pickle'
        self.min_accuracy = 0.987654321
        self.pickleop(mode='save')
        savedata = self.pickleop()
        self.print_log('savedata{}'.format(savedata))
    
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
    
    def data_process_end(self):
        '''
        数据处理结束要做的
        '''
        if self.args.local_debug:
            self.debug_print_var()
        
        self.test_pickle()
        pass
    
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
                print(self.quests)
            self.print_log('expend count:{}, {}, {}, {}'.format(quest_count, len(self.quests), test_count, len(self.testquests)))
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
        allquest = ''
        
#        for quest in self.quests:
#            allquest += quest
#        if self.args.get_data_method in [2]:
#            for quest in self.testquests:
#                allquest += quest
        
        for quest in (self.quests | self.testquests if self.args.get_data_method in [2] else self.quests):
            allquest += quest
        
        with open(self.file_path['chat'], 'r', encoding='utf-8') as f:#有木有	这个木有	这个没有]	这个没有 
            for line in f.readlines():
                allquest += line.upper().strip().split('\t')[0]#有木有 为何只有第一列
        if len(allquest) != 0:
            counter = Counter(allquest)
            count_pairs = counter.most_common(self.args.vocab_size - 1)
            words, _ = list(zip(*count_pairs))
            words = ['<UNK>'] + list(words)
        else:
            words = ['<UNK>']
        self.word_to_id = dict(zip(words, range(len(words))))#字分配递增ID(UNK为0号,最频繁的字分为1号,一直到最后)
        self.args.vocab_size = len(words)
    
    def get_labels(self):
        self.label_to_id = dict(zip(self.label_quest.keys(), range(len(self.label_quest))))#给label分配递增ID
        self.id_to_label = {str(v):k for k,v in self.label_to_id.items()}#为何要转换成str的ID??
        self.args.num_class = len(self.label_quest)#有多少类
    
    def extend_by_sameword(self,quest):         
        if 0 == self.args.use_same_words:#init的时候的同义词扩展
            if 1 == self.args.use_cut_words:
                return [self.word_parser.cut(quest)]
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
        if 1 == self.args.use_cut_words:
            quests = set([tuple(self.word_parser.cut(quest))])
        else:
            quests = set([quest])
        if 0 == len(local_sentences):
            return quests
        #遍历
        if 1 == self.args.use_cut_words:
            Allcut = []
            self.get_next_word(0, Allcut, None, local_sentences)
            index = 0
            while True:
                if index >= len(Allcut):
                    break
                steps = len(local_sentences)
                quests.add(tuple(Allcut[index:index+steps]))
                index += steps
        else:
            self.get_next_word(0, '', quests, local_sentences)
        return quests
    
    def get_next_word(self, index, quest, quests, local_sentences):#('如何开通银联在线支付', '如何付钱', '如何支付', '怎么付钱', '怎么支付', '怎样付钱', '怎样支付')
        if index == len(local_sentences):
            if 0 == self.args.use_cut_words:
                quests.add(quest)
            return              
        word = local_sentences[index] #[如何,怎么,怎样]
        if isinstance(word, list):
            for w in word:
                if 1 == self.args.use_cut_words:
                    quest.append(w)
                    self.get_next_word(index+1, quest, quests, local_sentences)#列表中的属于一个近义词
                else:
                    self.get_next_word(index+1, quest+w, quests, local_sentences)#列表中的属于一个近义词
        else:
            if 1 == self.args.use_cut_words:
                quest.append(word)
                self.get_next_word(index+1, quest, quests, local_sentences)#其他不适近义词的
            else:
                self.get_next_word(index+1, quest+word, quests, local_sentences)#其他不适近义词的
    
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
            vector=[]
            for x in data[i]:#data[i]每个问题,所以x是每个字
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
        quest = [self.word_to_id.get(word, self.word_to_id['<UNK>']) for word in raw_quest]
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