# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 17:59:39 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import sys
import os
import matplotlib.pyplot as plt
import configparser
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from collections import defaultdict
from prettytable import PrettyTable

def picture(accuracy):
    ymajorLocator = MultipleLocator(0.01) #将y轴主刻度标签设置为0.5的倍数
    ymajorLocator.MAXTICKS = 2000
    ymajorFormatter = FormatStrFormatter('%1.2f') #设置y轴标签文本的格式
    #yminorLocator  = MultipleLocator(0.001) #将此y轴次刻度标签设置为0.1的倍数
    #yminorLocator.MAXTICKS = 2000
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    picinfos = [[0, u'E_test', 'r^-'],
                [1, u'O_all', 'g^-'],
                [2, u'90%_chat', 'b^-'],
                [3, u'O_train', 'c^-'],
                [4, u'O_test', 'm^-']
            ]
    showTable = [0, 3, 4]
    xshow = range(len(accuracy))
    for info in picinfos:
        if info[0] not in showTable:
            continue
        yshow = []
        for index in range(len(accuracy)):
            variable = '{:04}'.format(index)
            yshow.append(accuracy[variable][0][info[0]])
        plt.plot(xshow, yshow, info[2], label=info[1])
        for (x, y) in zip(xshow, yshow):
            plt.text(x, y, '%.4f' % y, ha='center', va= 'bottom', fontsize=10)        
        
    #plt.ylim(0.0000, 1.0000)
    plt.legend(loc='upper left')
    plt.xlabel('times')
    plt.ylabel('accuracy')
    plt.title('the accuracy for compute')
    ax = plt.gca()
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_major_formatter(ymajorFormatter)
    #ax.yaxis.set_minor_locator(yminorLocator)
    #ax.xaxis.grid(True, which='major')
    #ax.yaxis.grid(True, which='minor')
    #ax.grid(True, linestyle='-.')
    #plt.xticks(x, group_labels, rotation=0) 
    plt.show()
    #plt.savefig('examples.jpg')

def summary(description):
    col_labels = ['no', 'model_name', 'result_vaild', 'c_model_name', 'c_result_vaild', 'use_same_words', 'use_same_file_class', 'get_data_method', 'get_batch_method']
    table1 = PrettyTable(col_labels)
    
    accuracy = defaultdict(list)
    for index in range(len(description)):
        variable = '{:04}'.format(index)
        accuracystr = description[variable].get_name()
        accuracy_sig = []
        startpos = 0
        endpos = 0
        while True:    
            startpos = accuracystr.find('{', startpos)
            endpos = accuracystr.find('}', endpos)
            if (-1 == startpos) or (-1 == endpos):
                break
            string = accuracystr[startpos+1: endpos]
            if 'NULL' == string:
                accuracy_sig.append(0.0000)
            else:
                accuracy_sig.append(float(string))
            startpos += 1
            endpos += 1
        if accuracystr not in accuracy:
            accuracy[variable].append(accuracy_sig)
        if '0000' == variable:
            continue
        summary_info = [index]
        model_names = description[variable].use_model__model_name
        summary_info.append(model_names)
        summary_info.append(description[variable].use_model__result_vaild)
        summary_info.append(description[variable].cut_word__model_name)
        summary_info.append(description[variable].cut_word__result_vaild)
        summary_info.append(description[variable].init_int__use_same_words)
        summary_info.append(description[variable].init_int__use_same_file_class)
        summary_info.append(description[variable].init_int__get_data_method)
        summary_info.append(description[variable].init_int__get_batch_method)
        table1.add_row(summary_info)
        
        model_names = model_names.split(',')
        for model_name in model_names:
            description[variable].readmodule(model_name.strip()[:-2])
            #要怎么处理 to do
        print(index)
        #        module_info = []
    with open('./procedure/report.txt', mode='w', encoding='utf-8') as fd:
        fd.write(str(table1))
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    ymajorLocator = MultipleLocator(0.01) #将y轴主刻度标签设置为0.5的倍数
    ymajorLocator.MAXTICKS = 2000
    ymajorFormatter = FormatStrFormatter('%1.2f') #设置y轴标签文本的格式
    picinfos = [[0, u'E_test', 'r^-'],
                [1, u'O_all', 'g^-'],
                [2, u'90%_chat', 'b^-'],
                [3, u'O_train', 'c^-'],
                [4, u'O_test', 'm^-']
            ]
    showTable = [0, 3, 4]
    xshow = range(len(accuracy))
    for info in picinfos:
        if info[0] not in showTable:
            continue
        yshow = []
        for index in range(len(accuracy)):
            variable = '{:04}'.format(index)
            yshow.append(accuracy[variable][0][info[0]])
        plt.plot(xshow, yshow, info[2], label=info[1])
        for (x, y) in zip(xshow, yshow):
            plt.text(x, y, '%.4f' % y, ha='center', va= 'bottom', fontsize=10)        
    plt.legend(loc='upper left')
    plt.xlabel('times')
    plt.ylabel('accuracy')
    plt.title('the accuracy for compute')
    ax = plt.gca()
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_major_formatter(ymajorFormatter)
    plt.ylim(0.0000, 1.0000)
    plt.savefig('./procedure/report.tiff')

class descript():
    def __init__(self, name):
        self.name = name
    
    def get_name(self):
        return self.name

    def readsummary(self, path):
        self.params = configparser.ConfigParser()
        self.params.read(path, encoding='utf-8')
        findMaps = [['use_model', 'model_name', 'str'],
                   ['use_model', 'result_vaild', 'str'],
                   ['cut_word', 'model_name', 'str'],
                   ['cut_word', 'result_vaild', 'str'],
                   ['init_int', 'use_same_words', 'int'],
                   ['init_int', 'use_same_file_class', 'int'],
                   ['init_int', 'get_data_method', 'int'],
                   ['init_int', 'get_batch_method', 'int']
                ]
        for item in findMaps:
            if self.isoptionexist(item[0], item[1]):
                setattr(self, item[0]+'__'+item[1], self.get_option(item[0], item[1], item[2]))
    
    def getAlloption(self, section):
        return self.params.items(section)
    
    def getAllsections(self):
        return self.params.sections()

    def readmodule(self, gettype='TextCNN'):
#        findMaps = [['summary', 'conv_maxpool_count', 'int'],
#                   ['summary', 'use_attention', 'int'],
#                   ['summary', 'attention_size', 'int'],
#                   ['summary', 'dropout_keep_prob', 'int'],
#                   ['summary', 'batch_size', 'int'],
#                   ['embedding', 'embedding_size', 'int']
#                ]
        for conftype, path in self.conf_filepaths.items():
            if gettype == conftype:
                self.params = configparser.ConfigParser()
                self.params.read(path, encoding='utf-8')
                all_sections = self.getAllsections()
                savesection = {}
                for section in all_sections:
                    savesection[section] = self.getAlloption(section)
                if gettype not in self.module_conf:
                    self.module_conf[gettype] = savesection
            else:
                continue

#    def readmodule(self):
#        findMaps = [['summary', 'lstm_size_each_layer', 'str'],
#                   ['summary', 'use_attention', 'int'],
#                   ['summary', 'attention_size', 'int'],
#                   ['summary', 'use_basic_cell', 'int'],
#                   ['summary', 'use_bidirectional', 'int'],
#                   ['summary', 'dropout_keep_prob', 'float'],
#                   ['summary', 'grad_clip', 'float'],
#                   ['summary', 'batch_size', 'int'],
#                   ['embedding', 'embedding_size', 'int']
#                ]
#        for conftype, path in self.conf_filepaths.items():
#            if 'LSTM' == conftype:
#                self.params = configparser.ConfigParser()
#                self.params.read(path, encoding='utf-8')
#            else:
#                continue
                
    def savemodule(self, path):
        self.module_conf = {}
        with open(path, mode='r', encoding='utf-8') as fd:
            contexts = fd.readlines()
        self.conf_filepaths = {}
        for context in contexts:
            if ((-1 != context.find('==>')) and (-1 != context.find('cfg'))):
                context = context.split('==>')
                endpos = context[1].strip().find('.cfg') + len('.cfg')
                self.conf_filepaths[context[0].strip()] = context[1].strip()[:endpos]
    
    def isoptionexist(self, section, option):
        return self.params.has_option(section, option)
    
    def get_option(self, section, option, wclass='str'):
        if 'str' == wclass:
            return self.params.get(section, option)
        elif 'bool' == wclass:
            return self.params.getboolean(section, option)
        elif 'int' == wclass:
            return self.params.getint(section, option)
        elif 'float' == wclass:
            return self.params.getfloat(section, option)

if __name__ == '__main__':
    #os.chdir(os.path.dirname(__file__))
    description = {}
    Path = sys.argv[0][:sys.argv[0].rfind('/')]
    split_bytes = len(Path) + 1
    for dirpath, dirnames, filenames in os.walk(Path):
        findonce = False
        if -1 != dirpath.find('0000'):
            description['0000'] = descript(dirpath[split_bytes:])
        tmpdescript = None
        for file in filenames:
            if file.endswith('.cfg'):
                reportname = dirpath[split_bytes:]
                description[reportname[:4]] = descript(reportname)
                tmpdescript = description[reportname[:4]]
                tmpdescript.readsummary(os.path.join(dirpath, file).replace('\\', '/'))
                findonce = True
            if findonce and file.endswith('.txt'):
                tmpdescript.savemodule(os.path.join(dirpath, file).replace('\\', '/'))
                findonce = False
    summary(description)
    
#    top = tkinter.Tk()
#    im=Image.open('examples.jpg')
#    img=ImageTk.PhotoImage(im)
#    imLabel=tkinter.Label(top,image=img).pack()
    
#    load = Image.open('examples.jpg')
#    img_png = tkinter.PhotoImage(load)
#    top.mainloop()