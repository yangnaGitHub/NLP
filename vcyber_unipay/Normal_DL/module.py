# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 18:29:13 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import tensorflow as tf
import os
import traceback

class NormalDL():
    def __init__(self, args, params, log=None, **autodict):
        os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
        self.args = args
        self.params = params
        self.log = log
        self.autodict = autodict
        self.construct()
    
    def print_log(self, message):
        if self.args.local_debug:
            print('[%s][%s:%s]' % (os.path.basename(__file__), self.__class__.__name__, traceback.extract_stack()[-2][2]), message)
        if self.args.print_to_log:
            if self.log:
                self.log.print_to_file(message)
    
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
            try:
                return self.get_option(section, name, wclass)
            except Exception as e:
                self.print_log(e)
    
    def get_shape(self, var):
        return tf.shape(var)
    
    def construct(self):
        lastsize = self.get_params('summary', 'input_size', 'int')
        with tf.name_scope('input'):
            self.input_x = tf.placeholder(tf.float32, [None, lastsize], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, self.args.num_class], name="input_y")
            self.keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")#正则化
            #self.print_log('input_x.shape{}'.format(self.input_x.shape))
        
#        activate_fun_map = {
#                'sigmoid':'sigmoid',
#                'tanh':'tanh',
#                'relu':'nn.relu',
#                'relu6':'nn.relu6',
#                'elu':'nn.elu',
#                'softplus':'nn.softplus',
#                'softsign':'nn.softsign',
#                }
        
        with tf.name_scope('layer'):
            self.r_hidden_0 = self.input_x
            hidden_layers = self.get_option('summary', 'hidden_layer', 'str').split(',')
            activate_function = self.get_option('summary', 'activate_function', 'str')
            hidden_lists = []
            for index, hidden in enumerate(hidden_layers):
                cellcount = int(hidden.strip())
                w_name = 'w_hidden_{}'.format(index + 1)
                b_name = 'b_hidden_{}'.format(index + 1)
                o_name = 'o_hidden_{}'.format(index + 1)
                r_name = 'r_hidden_{}'.format(index + 1)
                r_name_last = 'r_hidden_{}'.format(index)
                hidden_lists.append(tf.Variable(tf.truncated_normal([lastsize, cellcount], stddev=0.1), name=w_name))
                hidden_lists.append(tf.Variable(tf.zeros([cellcount]), name=b_name))
                setattr(self, o_name, tf.matmul(eval('self.' + r_name_last), hidden_lists[2*index]) + hidden_lists[2*index + 1])
                setattr(self, r_name, tf.nn.dropout(eval('tf.nn.' + activate_function + '(self.' + o_name + ')'), self.keep_prob))
                #setattr(self, o_name, tf.matmul(eval('self.' + r_name_last), tf.Variable(tf.truncated_normal([lastsize, cellcount], stddev=0.1), name=w_name)) + tf.Variable(tf.zeros([cellcount]), name=b_name))
                lastsize = cellcount
            
            l2_loss = tf.constant(0.0)
            for hidden_list in hidden_lists:
                l2_loss += tf.nn.l2_loss(hidden_list)
        
        with tf.name_scope('output'):
            #setattr(self, o_name, tf.matmul(eval('self.' + r_name_last), 
            w_output = tf.Variable(tf.truncated_normal([lastsize, self.args.num_class], stddev=0.1), name='w_output')
            b_output = tf.Variable(tf.zeros([self.args.num_class]), name='b_output')
            l2_loss += tf.nn.l2_loss(w_output)
            l2_loss += tf.nn.l2_loss(b_output)
            o_output = tf.matmul(eval('self.' + r_name), w_output) + b_output
            #self.r_output = eval('tf.nn.' + activate_function + '(' + o_output + ', name=r_output)')
            self.scores = tf.nn.softmax(o_output)
            self.predictions = tf.argmax(o_output, 1, name="predictions")
        
        with tf.name_scope("accuracy"):
            #self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.input_y, axis=1), self.predictions), tf.float32))
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))#准确率
            self.acc = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
        
        with tf.name_scope('train'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=o_output, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + float(self.get_params('summary', 'l2_reg_lambda')) * l2_loss
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.get_option('summary', 'learning_rate', 'float')).minimize(self.loss)