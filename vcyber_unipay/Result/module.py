# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 18:29:13 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import tensorflow as tf
import os
import traceback

class NormalML():
    def __init__(self, args, params, log=None):
        pass
    
    def print_log(self, message):
        if self.args.local_debug:
            print('[%s][%s:%s]' % (os.path.basename(__file__), self.__class__.__name__, traceback.extract_stack()[-2][2]), message)
        if self.args.print_to_log:
            if self.log:
                self.log.print_to_file(message)
    
    def get_shape(self, var):
        return tf.shape(var)
    
    def construct(self):
        self.input_x = tf.placeholder(tf.float32, [None, self.args.num_class], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.args.num_class], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")#正则化
        l2_loss = tf.constant(0.0)
        
        attention_w = tf.Variable(tf.truncated_normal([num_filters_total, attention_size], stddev=0.1), name='attention_w')
        attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
        