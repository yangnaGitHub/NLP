# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 10:06:25 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import jieba

class TrieNode(object):
    def __init__(self):
        self.data = {}
        self.is_word = False

class Trie():
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for letter in word:
            child = node.data.get(letter)
            if not child:
                node.data[letter] = TrieNode()
#            if False == node.is_word:#natasha觉得是不是应该放在这边 ==> 不该放在这边,就该放在外面,词的最后一个字的时候设置起来(判断是一个完整的词)
#                node.is_word = True
            node = node.data[letter]
        node.is_word = True
        
    def search(self, word):
        node = self.root
        for letter in word:
            node = node.data.get(letter)
            if not node:
                return False
        return node.is_word  # 判断单词是否是完整的存在在trie树中
    
    def cut(self, sentence):
        sentence = list(sentence)
        word = ''
        new_sentence = []
        i = 0

        while(i < len(sentence)):
            bfind = False
            j = len(sentence)
            #对于当前的每一个字，从后到前找
            while(i <= j):
                word = ''.join(sentence[i:j])
                #print(i, j, ":", word)
                if self.search(word):
                    new_sentence.append(word)
                    #找到了i就往前移动
                    i += len(word)
                    bfind = True
                    break
                else:
                    #没有找到，j往前移动
                    j -= 1
            if bfind == False:
                #从头到尾都没找到，则i向前移动一步
                new_sentence.append(sentence[i])
                i += 1
        return new_sentence
    

class TrieTree(object):
    def __init__(self, data):
        self.data = data
        self.TrieRoot = Trie()
        self.ConstructTree()
    
    def ConstructTree(self):
        #同义词表的字典构建,这个字典就是用来分词出来后用作问题的扩展的
        for name in self.data.get_samewords_filename():
            with open(name, 'r', encoding='utf-8') as f:#../data/common.txt ../data/bank.txt
                for line in f.readlines():
                    line = line.upper().strip().split('=')#英文字母的处理 A0001=能;可以
                    if len(line) <= 1:#数据格式错误不处理
                        continue
                    words = line[1].strip().split(";")#以分号分块 能,可以
                    if len(words) > 0:
                        [self.TrieRoot.insert(word) for word in words]
        #分词分错 必定会影响扩展问题 就影响输入问题集合
        
        #针对有歧义的分词 如果有同义词 同义词替换后扩展出来的所有的问题集 比较问题集看谁更合理 ==> 应该能提高扩展问题的合理性,但是谁更合理的计算指标要怎么处理呢
         #更合理的计算指标==>应该能提升分词的准确率

        #普通字典的字典树构建 <== 分词出现更多可能性 ==> 会不会提高扩展问题的合理性有待商榷
         #比如说分到的是同义词表中的词对原问题的扩展不影响
         #如果分到的是新加的普通字典,没有同义词表所以不对原问题扩展
          #如果分词是正确的,那么避免分到不正确并且因为扩展问题的缘故使得错误不至于加深
          #如果分词是错误的,那么可能会本来可以分词正确的问题分词错误同时也没有扩展
        
    def cut(self, sentence):
        return self.TrieRoot.cut(sentence)

class JieBa(object):
    def __init__(self, data):
        pass
    
    def cut(self, sentence):
        return list(jieba.cut(sentence, cut_all=False))

class word_parser(object):
    def __init__(self, data):
        self.data = data
        self.args = data.args
        self.save_methods = {}
        self.make_mothod()
        for parser_method in self.parser_methods:
            self.save_methods[parser_method] = eval(parser_method + '(self.data)')
        #其他的分词方法
    
    def make_mothod(self):##有几种切词方式,使用这种方式好扩展
        section = 'cut_word'
        model_name_str = 'model_name'
        result_vaild_str = 'result_vaild'
        self.parser_methods = []
        self.parser_methods_Vaild = []
        if self.args.isoptionexist(section, model_name_str):
            model_names = self.args.get_option(section, model_name_str, 'str')
            for name in model_names.split(','):
                self.parser_methods.append(name.strip())
        else:
            self.print_log('model_name not exist')
        if self.args.isoptionexist(section, result_vaild_str):
            result_vaild = self.args.get_option(section, result_vaild_str, 'str')
            for name in result_vaild.split(','):
                self.parser_methods_Vaild.append(name.strip())
    
    def cut(self, sentence):
        self.cut_result = {}
        for key, value in self.save_methods.items():
            if key in self.parser_methods_Vaild:
                self.cut_result[key] = value.cut(sentence)
        #综合所有的分词最后得到最合适的分词,是不是要这么做
        #end
        return self.cut_result[self.parser_methods_Vaild[0]]