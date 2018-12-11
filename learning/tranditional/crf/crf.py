# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 08:14:00 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import numpy as np
import codecs
import re
import os
import copy

class crf():
    def __init__(self, trainfile, templatefile):
        self.trainfile = trainfile
        self.templatefile = templatefile
        self.data_vars = ['text_data', 'text_label', 'label_to_id', 'id_to_label']
    
    def __template(self):
        if not hasattr(self, 'template'):
            self.template = {}
        fd_contents = codecs.open(self.templatefile)
        for content in fd_contents:
            content = content.strip()
            if 0 == len(content):#空行
                continue
            if content.startswith('#'):#注释行
                continue
            if (ord(content[0]) | 0x20) not in [ord('u'), ord('b')]:#大写变小写 => int变chr + unichr
                continue
            if content.count('[') != content.count(']'):
                continue
            
            content = content.split(':')
            if 2 != len(content):
                continue
            
            search_lists = re.findall('\[-?\d+,-?\d+\]', content[1])
            self.template[content[0]] = [[int(tag) for tag in re.findall('\[(.*)?\]', search)[0].split(',')] for search in search_lists]
        print(self.template)
    
    def __data(self, content_size=3):
        for var in self.data_vars:
            if not hasattr(self, var):
                setattr(self, var, {})
        fd_contents = codecs.open(self.trainfile)
        
        block_id = 0
        block_data = []
        block_label = []
        for content in fd_contents:
            content = content.strip()
            if 0 == len(content):#空行 连续行
                if block_data and block_label:
                    self.text_data[block_id] = copy.deepcopy(block_data)
                    self.text_label[block_id] = copy.deepcopy(block_label)
                    block_data.clear()
                    block_label.clear()
                    block_id += 1
            else:
                content = content.split(' ')
                if content_size != len(content):
                    continue
                block_data.append(content[0:-1])
                if not self.label_to_id:
                    self.label_to_id[content[-1]] = 0
                else:
                    if content[-1] not in self.label_to_id:
                        self.label_to_id[content[-1]] = max(list(self.label_to_id.values())) + 1
                block_label.append(self.label_to_id[content[-1]])
        #end
        self.text_data[block_id] = block_data
        self.text_label[block_id] = block_label
        #self.id_to_label
        self.id_to_label.update(dict([(val, key) for key, val in self.label_to_id.items()]))
        print(self.text_data)
        print(self.text_label)
        print(self.label_to_id)
        print(self.id_to_label)
    
    def __process_feature(self, threshold):
        if not hasattr(self, 'u_features'):
            setattr(self, 'u_features', {})
        if not hasattr(self, 'b_features'):
            setattr(self, 'b_features', {})

        u_features = {}
        b_features = {}
        def generate_feature(val_text_data):

            for index, val_text in enumerate(val_text_data):
                for key_template, val_template in self.template.items():
                    offsets = [(index+int(val[0]), int(val[1])) for val in val_template if index+int(val[0]) in range(len(val_text_data)) if int(val[1]) in range(len(val_text))]
                    string = key_template
                    for offset in offsets:
                        string += ":" + val_text_data[offset[0]][offset[1]]
                    if ord('u') == (ord(key_template[0]) | 0x20):
                        if string in u_features:
                            u_features[string] += 1
                        else:
                            u_features[string] = 1
                    if ord('b') == (ord(key_template[0]) | 0x20):
                        if string in b_features:
                            b_features[string] += 1
                        else:
                            b_features[string] = 1

        for key_text_data, val_text_data in self.text_data.items():
            generate_feature(val_text_data)
        
        #threshold
        if threshold > 1:
            self.u_features = {key:val for key,val in u_features.items() if val >= threshold}
            self.b_features = {key:val for key,val in b_features.items() if val >= threshold}
        
    
    def train(self):
        self.__template()
        self.__data()
        self.__process_feature()
        

if __name__ == '__main__':
    if os.getcwd() != os.path.dirname(__file__):
        os.chdir(os.path.dirname(__file__))
    crf_test = crf('train.data', 'template.txt')
    crf_test.train()