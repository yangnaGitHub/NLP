# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:13:59 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import logging
import time
import os
from threading import Timer

class log():
    def __init__(self, args):
        self.args = args
        self.dirpath = self.args.print_log_path
        self.firstdate = time.strftime("%Y%m%d")
        self.prepare()
    
    def prepare(self):        
        if not os.path.exists(self.dirpath + '/log'):
            os.makedirs(self.dirpath + '/log')
        self.filename = self.dirpath + '/log/' + self.firstdate + '.txt'
    
    def create_new_file(self, nowdate):
        self.filename = self.dirpath + '/log/' + nowdate + '.txt'
    
    def print_to_file(self, info, level='nolevel'):
        nowdate = time.strftime("%Y%m%d")#每一天都有log
        if nowdate != self.firstdate:
            self.create_new_file(nowdate)
        
        if 'nolevel' == level:
            if self.args.isoptionexist('init_str', 'print_log_level'):
                level = self.args.print_log_level
            else:
                level = 'debug'
            
        loc_time = time.strftime("%H:%M:%S")
        if 'debug' == level:
            logging.basicConfig(filename=self.filename, level=logging.DEBUG)
            logging.debug('{}:{}'.format(loc_time, info))
        else:
            logging.basicConfig(filename=self.filename, level=logging.ERROR)
            logging.error('{}:{}'.format(loc_time, info))

    #启动一个守护线程
    #Timer定时器:隔一定时间调用一个函数,如果想实现每隔一段时间就调用一个函数的话,就要在Timer调用的函数中,再次设置Timer(Timer是Thread的一个派生类)
    def log_deamon(self):
        loc_time = time.strftime("%H")
        if '24' == loc_time:
            log('守护线程信息,请忽略')
        t = Timer(3600, self.log_deamon)#3600S后
        t.start()#一个小时后再启动
