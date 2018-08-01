# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 17:40:55 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import tensorflow as tf
import os
import traceback
import configparser

class Embed(object):
    def __init__(self, args, data, log=None, **autodict):
        os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
        self.module_name = 'Embed'
        self.model = None
        self.args = args
        self.data = data
        self.log = log
        self.autodict = autodict
        self.params = configparser.ConfigParser()
        if self.args.isoptionexist('use_model', 'module_conf'):
            module_conf = self.args.get_option('use_model', 'module_conf', 'str')
        else:
            module_conf = 'module.cfg'
        self.params.read('./' + self.module_name + '/' + module_conf)
        #self.params.read('./' + 'module.cfg')
#        if 'use_old_model' in autodict:
#            self.old_path = autodict['use_old_model']
#        else:    
#            self.construct()
    
    def get_option(self, section, option, wclass='str'):
        if 'str' == wclass:
            return self.params.get(section, option)
        elif 'bool' == wclass:
            return self.params.getboolean(section, option)
        elif 'int' == wclass:
            return self.params.getint(section, option)
        elif 'float' == wclass:
            return self.params.getfloat(section, option)
    
    def get_params(self, section='summary', name='skip_window', wclass='str'):
        if name in self.autodict:
            return self.autodict[name]
        else:
            return self.get_option(section, name, wclass)
    
    def print_log(self, message):
        if self.args.local_debug:
            print('[%s][%s:%s]' % (os.path.basename(__file__), self.__class__.__name__, traceback.extract_stack()[-2][2]), message)
        if self.args.print_to_log:
            if self.log:
                self.log.print_to_file(message)
    
    def construct(self):
        self.input_x = tf.placeholder(tf.int32, [None], name="input_x")#batch
        self.input_y = tf.placeholder(tf.int32, [None, self.get_params(name='window_size', wclass='int')], name="input_x")#batch*skip_windows
        self.print_log('self.input_y:{}'.format(self.input_y.shape))
        
        with tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([self.args.vocab_size, self.get_params(section='embedding', name='embedding_size', wclass='int')], -1, 1), name='W')
            self.embedding = tf.nn.embedding_lookup(self.W, self.input_x)
            self.print_log('self.embedding:{}'.format(self.embedding.shape))
        
        with tf.name_scope('loss'):
            softmax_w = tf.Variable(tf.truncated_normal([self.args.vocab_size, self.get_params(section='embedding', name='embedding_size', wclass='int')], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(self.args.vocab_size))
            loss = tf.nn.nce_loss(softmax_w, softmax_b, self.input_y, self.embedding, self.get_params(name='n_sampled', wclass='int'), self.args.vocab_size)#n_sampled采样出多少个负样本
            self.print_log('self.loss:{}'.format(loss.shape))
            self.cost= tf.reduce_mean(loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.get_params(name='learning_rate', wclass='float')).minimize(self.cost)
    
#class test():
#    def __init__(self):
#        self.vocab_size = 1000
#        self.local_debug = 1
#        self.print_to_log = 0
#
#if __name__ == '__main__':
#    os.chdir(os.path.dirname(__file__))
#    signal = Embed(test(), None, None)