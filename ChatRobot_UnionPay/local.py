# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:35:22 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import Allmodule

class Local():
    def __init__(self, args):
        self.args = args
        self.module = Allmodule.module(self.args, None)
    
    def debug(self):
        if self.args.isoptionexist('local_params', 'local_method_test'):
            method = self.args.get_option('local_params', 'local_method_test', 'str')
        else:
            method = 'train'

        if 'train' == method:
            self.module.train()
        elif 'chat' == method:
            self.module.predict()
        elif 'lookup' == method:
            self.module.lookup()
        elif 'log' == method:
           self.module.start_log()