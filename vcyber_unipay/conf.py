# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:13:59 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import configparser

class myargs:
    def __init__(self, filepath='./Allmodule.cfg'):
#        import os
#        os.chdir('E:\\AboutStudy\\code\\python\\natasha_suda')
        self.version = '3.0.0'
        self.params = configparser.ConfigParser()
        self.params.read(filepath, encoding='utf-8')
        self.initfile()
        
    def get_option(self, section, option, wclass='str'):
        if 'str' == wclass:
            return self.params.get(section, option)
        elif 'bool' == wclass:
            return self.params.getboolean(section, option)
        elif 'int' == wclass:
            return self.params.getint(section, option)
        elif 'float' == wclass:
            return self.params.getfloat(section, option)
    
    def initfile(self):
        all_sections = self.getAllsections()
        wclass = ['str', 'bool', 'float', 'int']
        for section in all_sections:
            for sclass in wclass:
                optionname = 'init_' + sclass
                if (self.params.has_option(section, optionname)) and (self.get_option(section, optionname, 'bool')):
                    self.initsection(section, sclass)
    
    def isoptionexist(self, section, option):
        return self.params.has_option(section, option)
    
    def issectionexist(self, section):
        return self.params.has_section(section)

    def getAlloption(self, section):
        return self.params.items(section)
    
    def initsection(self, section, wclass):
        alloptions = self.getAlloption(section)
        for option in alloptions:
            if ('init_' + wclass) == option[0]:
                continue
            setattr(self, option[0], self.get_option(section, option[0], wclass))#eval('self.' + option) = self.get_option(section, option, wclass)
    
    def getAllsections(self):
        return self.params.sections()
    
#    def update(self, argc, argv):
#        params_mapping = {
#                '-p':'http_port',
#                '-ho':'http_host',
#                '-u':'user_id',
#                '-s':'use_same_words',
#                '-lo':'print_to_log',
#                '-i':'use_same_file_class',
#                '--version':'version',
#                }
#        for index in range(1, argc):
#            pass

args = myargs()
#print(args.getAlloption('data_pickle'))

#signal module test  
#if __name__ == '__main__':
#    if os.getcwd() != os.path.dirname(__file__):
#        os.chdir(os.path.dirname(__file__))
#    args = myargs()
#    test = args.getAlloption('init_str')
#    for index in test:
#        print(type(index), index)