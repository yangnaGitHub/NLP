# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 11:37:30 2019

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

#分词方法,逆向最大匹配
class RMM():
    def __init__(self, dict_path):
        #prepare dictionary
        self.dictionary = set()
        self.maximum = 0
        with open(dict_path, 'r', encoding='utf-8') as fd:
            for line in fd:
                line = line.strip()
                if not line:
                    continue
                self.dictionary.add(line)
                self.maximum = max(self.maximum, len(line))
    
    def cut(self, cut_str):
        result = []
        index = len(cut_str)
        while 0 < index:
            findword = False
            for steptag in range(self.maximum, 0, -1):
                step = index - steptag
                if 0 > step:
                    step = 0
                temp = cut_str[step:index]
                if temp in self.dictionary:
                    findword = True
                    result.append(temp)
                    index = step
                    break
            if not findword:
                result.append(cut_str[index-1])
                index -= 1
        return result