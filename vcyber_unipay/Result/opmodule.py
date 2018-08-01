# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 18:28:43 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import tensorflow as tf
from Result.module import NormalML as NormalML
from datetime import timedelta
import os
import traceback
import time
import numpy as np

class ResultOp():
    def __init__(self, args, params, log=None):
        pass
    
    def print_log(self, message):
        if self.args.local_debug:
            print('[%s][%s:%s]' % (os.path.basename(__file__), self.__class__.__name__, traceback.extract_stack()[-2][2]), message)
        if self.args.print_to_log:
            if self.log:
                self.log.print_to_file(message)
    
    def get_time_dif(self, start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))
    
    def print_shape(self, var, feed_dict=None, prefix='shape'):
        self.print_log('{}==>{}'.format(prefix, self.session.run(self.model.get_shape(var), feed_dict=feed_dict)))
    
    def train(self):
        pass
    
    def load_model(self):
        with tf.Graph().as_default() as g:
            self.model = NormalML(self.args, self.params)
            self.session = tf.Session(graph=g)
            with self.session.as_default():
                self.session.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                saver.restore(sess=self.session, save_path=self.module_path)