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
import operator

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

def check_the_conf(src, des):
    diff_conf = []
    for key in src.keys():
        if key not in des:
            #diff_conf.extend(src[key])
            diff_conf.extend([(key+'__'+item[0], item[1]) for item in src[key]])
        else:
            if not operator.eq(list(src[key]), list(des[key])):
                des_value = [value[0] for index, value in enumerate(des[key])]
                for value in src[key]:
                    if value[0] not in des_value:
                        diff_conf.append(value)
                    else:
                        if value[1] != des[key][des_value.index(value[0])][1]:
                            diff_conf.append(value)
    return diff_conf

def summary(description):
    col_labels = ['no', 'model_name', 'result_vaild', 'c_model_name', 'c_result_vaild', 'use_same_words', 'use_same_file_class', 'get_data_method', 'get_batch_method', 'add_outside_sets']
    table1 = PrettyTable(col_labels)
    accuracy = defaultdict(list)
    
    model_conf_base = {}
    diff_conf = {}
    base_diff = {}
    for index in range(len(description)):
        variable = '{:04}'.format(index)
        #picture
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
        #allmodule.cfg
        summary_info = [index]
        model_names = description[variable].use_model__model_name
        summary_info.append(model_names)
        summary_info.append(description[variable].use_model__result_vaild)
        summary_info.append(description[variable].cut_word__model_name)
        summary_info.append(description[variable].cut_word__result_vaild)
        
        if 1 == description[variable].init_int__use_same_words:
            summary_info.append('use')
        else:
            summary_info.append('not use')
        
        tempvalue = description[variable].init_int__use_same_file_class
        origin_string = 'common'
        if 1 == tempvalue:
            origin_string += (' + car')
        elif 2 == tempvalue:
            origin_string += (' + bank')
        summary_info.append(origin_string)
        
        #summary_info.append(description[variable].init_int__use_same_words)
        #summary_info.append(description[variable].init_int__use_same_file_class)
        summary_info.append(description[variable].init_int__get_data_method)
        summary_info.append(description[variable].init_int__get_batch_method)
        
        tempvalue = description[variable].init_int__add_outside_sets
        if -1 == tempvalue:
            origin_string = 'not definite'
        elif 0 == tempvalue:
            origin_string = 'not add outsides'
        elif 1 == tempvalue:
            origin_string = 'add outsides'
        summary_info.append(origin_string)
        #summary_info.append(description[variable].init_int__add_outside_sets)
        table1.add_row(summary_info)
        
        model_names = model_names.split(',')
        for model_name in model_names:
            description[variable].readmodule(model_name.strip()[:-2])
            current_conf = description[variable].module_conf[model_name.strip()[:-2]]
            #要怎么处理 to do
#            counts = [(key, len(value)) for key, value in current_conf.items()]
#            totalcount = sum([count[1] for count in counts])

            if model_name in model_conf_base.keys():
                diff_conf[model_name][index] = {}
                if operator.eq(list(model_conf_base[model_name].keys()), list(current_conf.keys())):
                    if not operator.eq(list(model_conf_base[model_name].values()), list(current_conf.values())):
                        #双向比较
                        diff_conf[model_name][index]['base'] = check_the_conf(model_conf_base[model_name], current_conf)
                        diff_conf[model_name][index]['diff'] = check_the_conf(current_conf, model_conf_base[model_name])
                else:
                    diff_conf[model_name][index]['base'] = check_the_conf(model_conf_base[model_name], current_conf)
                    diff_conf[model_name][index]['diff'] = check_the_conf(current_conf, model_conf_base[model_name])
            else:
                model_conf_base[model_name] = current_conf
                base_diff[model_name] = index
                diff_conf[model_name] = {index:{}}
    #save allmodule.cfg
    with open('./procedure/report.txt', mode='w', encoding='utf-8') as fd:
        fd.write('summary:\n' + str(table1) + '\n')
    
    #deal module.cfg
    omit_table = ['module_path', 'pickle_file', 'module_file', 'embed_file']
    for key, value in diff_conf.items():
        for subkey, subvalue in value.items():
            col_labels = ['no']
            if subvalue:
                base_list = [base_diff[key]]
                for item in subvalue['base']:
                    if item[0] not in omit_table:
                        col_labels.append(item[0])
                        base_list.append(item[1])
                diff_list = ['not definite'] * len(base_list)
                diff_list[0] = subkey
#                print(col_labels)
#                print(base_list)
#                print(diff_list)
                for item in subvalue['diff']:
                    if item[0] not in omit_table:
                        if item[0] not in col_labels:
                            col_labels.append(item[0])
                            diff_list.append(item[1])
                            base_list.append('not definite')
                        else:
                            diff_list[col_labels.index(item[0])] = item[1]
                table = PrettyTable(col_labels)
                table.add_row(base_list)
                table.add_row(diff_list)
                with open('./procedure/report.txt', mode='a', encoding='utf-8') as fd:
                    fd.write('\n' + str(table) + '\n')
            else:
                with open('./procedure/report.txt', mode='a', encoding='utf-8') as fd:
                    fd.write('\n\n' + str(key) + ':')
  
    #draw picture
    #%matplotlib qt5
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    ymajorLocator = MultipleLocator(0.01) #将y轴主刻度标签设置为0.5的倍数
    ymajorLocator.MAXTICKS = 2000
    ymajorFormatter = FormatStrFormatter('%1.2f') #设置y轴标签文本的格式
    xmajorLocator = MultipleLocator(1)
    picinfos = [[0, u'E_test', 'r^-', u'Ex_test'],
                [1, u'O_all', 'g^-', u'Ori_all'],
                [2, u'90%_chat', 'b^-', u'90%_chat'],
                [3, u'O_test', 'c^-', u'Ori_test'],
                [4, u'O_train', 'm^-', u'Ori_train']
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
        plt.plot(xshow, yshow, info[2], label=info[3])
        for (x, y) in zip(xshow, yshow):
            plt.text(x, y, '%.4f' % y, ha='center', va= 'bottom', fontsize=10)        
    plt.legend(loc='upper left')
    plt.xlabel('times')
    plt.ylabel('accuracy')
    plt.title('the accuracy for compute')
    ax = plt.gca()
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_major_formatter(ymajorFormatter)
    ax.xaxis.set_major_locator(xmajorLocator)
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
                   ['init_int', 'get_batch_method', 'int'],
                   ['init_int', 'add_outside_sets', 'int']
                ]
        for item in findMaps:
            if self.isoptionexist(item[0], item[1]):
                setattr(self, item[0]+'__'+item[1], self.get_option(item[0], item[1], item[2]))
            else:
                if 'int' == item[2]:
                    setattr(self, item[0]+'__'+item[1], -1)
                elif 'float' == item[2]:
                    setattr(self, item[0]+'__'+item[1], -1.0)
                elif 'str' == item[2]:
                    setattr(self, item[0]+'__'+item[1], 'NULL')
                elif 'bool' == item[2]:
                    setattr(self, item[0]+'__'+item[1], False)
    
    def getAlloption(self, section):
        return self.params.items(section)
    
    def getAllsections(self):
        return self.params.sections()

    def readmodule(self, gettype='TextCNN'):
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
                #print(self.module_conf)
            else:
                continue
                
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
            reportname = dirpath[split_bytes:]
            if reportname.startswith('0000'):
                description['0000'] = descript(reportname)
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