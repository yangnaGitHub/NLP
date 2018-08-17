# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 15:40:36 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import configparser
import os

class myargs():
    def __init__(self, filepath='./conf/conf.cfg'):
        os.chdir('E:\\AboutStudy\\code\\python\\DRL')
        self.version = '3.0.0'
        self.params = configparser.ConfigParser()
        self.params.read(filepath, encoding='utf-8')
        
    def get_option(self, section, option, wclass='str'):
        if 'str' == wclass:
            return self.params.get(section, option)
        elif 'bool' == wclass:
            return self.params.getboolean(section, option)
        elif 'int' == wclass:
            return self.params.getint(section, option)
        elif 'float' == wclass:
            return self.params.getfloat(section, option)
    
    def isoptionexist(self, section, option):
        return self.params.has_option(section, option)
    
    def issectionexist(self, section):
        return self.params.has_section(section)

    def getAlloption(self, section):
        return self.params.items(section)
    
    def getAllsections(self):
        return self.params.sections()
    
    def listAlloptions(self):
        allsections = self.getAllsections()
        print([self.getAlloption(section) for section in allsections])

args = myargs()