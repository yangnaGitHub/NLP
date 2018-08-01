# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:03:23 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import tensorflow as tf
import os
import traceback
import re

class Lstm(object):
    def __init__(self, args, params, log=None):
        os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
        self.args = args
        self.params = params
        self.log = log
        self.construct()
    
    def get_option(self, section, option, wclass='str'):
        if 'str' == wclass:
            return self.params.get(section, option)
        elif 'bool' == wclass:
            return self.params.getboolean(section, option)
        elif 'int' == wclass:
            return self.params.getint(section, option)
        elif 'float' == wclass:
            return self.params.getfloat(section, option)

    def print_log(self, message):
        if self.args.local_debug:
            print('[%s][%s:%s]' % (os.path.basename(__file__), self.__class__.__name__, traceback.extract_stack()[-2][2]), message)
        if self.args.print_to_log:
            if self.log:
                self.log.print_to_file(message)
    
    def construct_model(self, lstm_size_each_layer):
        use_basic_cell = self.get_option('summary', 'use_basic_cell', 'int')
        if 0 == use_basic_cell:
            #lstm_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(int(size.strip()), state_is_tuple=False), output_keep_prob=self.keep_prob) for size in lstm_size_each_layer])
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(int(size.strip())), output_keep_prob=self.keep_prob) for size in lstm_size_each_layer])
        elif 1 == use_basic_cell:
            #lstm_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(int(size.strip()), state_is_tuple=False), output_keep_prob=self.keep_prob) for size in lstm_size_each_layer])
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(int(size.strip())), output_keep_prob=self.keep_prob) for size in lstm_size_each_layer])
        elif 2 == use_basic_cell:
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(int(size.strip())), output_keep_prob=self.keep_prob) for size in lstm_size_each_layer])
        return lstm_cell
    
    def get_shape(self, var):
        return tf.shape(var)
        
    def construct(self):
        tf.set_random_seed(66)
        
        #构建模型
        self.input_x = tf.placeholder(tf.int32, [None, self.args.max_document_lenth], name="input_x")#这是要一次训练None * self.args.max_document_lenth
        self.input_y = tf.placeholder(tf.float32, [None, self.args.num_class], name="input_y")#所有的分类,要不要多加一类??都没有命中(get_labels中确定class的数值)
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")#正则化
        
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.W = tf.Variable(tf.truncated_normal([self.args.vocab_size, self.get_option('embedding', 'embedding_size', 'int')], stddev=0.1), name='W')
            self.embedding = tf.nn.embedding_lookup(self.W, self.input_x)#(batch_size, max_document_lenth, embedding_size)
        self.print_log('embedding:{}'.format(self.embedding.shape))
        #model
        with tf.variable_scope('layer'):
            
            lstm_size_each_layer = self.get_option('summary', 'lstm_size_each_layer', 'str').split(',')
            use_bidirectional = self.get_option('summary', 'use_bidirectional', 'int')
            self.lstm_cell = self.construct_model(lstm_size_each_layer)
            #看shape
#            self.print_log('lstm_cell:{}, {}'.format(self.lstm_cell.input_shape, self.lstm_cell.output_shape))
#            self.print_log('lstm_cell:{}'.format(self.lstm_cell.state_size()))#(?, 122, lstm_size_each_layer[-1])
#            self.lstm_cell = tf.nn.rnn_cell.LSTMCell(128, state_is_tuple=False)
#            initial_state_fw = self.lstm_cell.zero_state(32, dtype=tf.float32)
#            self.print_log('lstm_cell:{}, {}, {}, {}'.format(len(initial_state_fw), initial_state_fw[0].shape, initial_state_fw[1].shape, initial_state_fw[2].shape))
#            self.print_log('lstm_cell:{}'.format(initial_state_fw[0].shape))
            
            if use_bidirectional:
                #output_fw:[batch_size, max_time, cell_fw.output_size] + output_bw:[batch_size, max_time, cell_bw.output_size]
                self.lstm_cell_bk = self.construct_model(lstm_size_each_layer)
                self.outputs, self.output_states = tf.nn.bidirectional_dynamic_rnn(self.lstm_cell, self.lstm_cell_bk, self.embedding, dtype=tf.float32)
                self.print_log('outputs:{},{}'.format(self.outputs[0].shape, self.outputs[1].shape))
            else:
                self.outputs, self.output_states = tf.nn.dynamic_rnn(self.lstm_cell, self.embedding, dtype=tf.float32)#[batch_size, max_time, cell.output_size]
                self.print_log('outputs:{}'.format(self.outputs.shape))#(?, 122, lstm_size_each_layer[-1])
                #self.print_log('outputs:{}, {}, {}, {}, {}'.format(self.outputs.shape, len(self.output_states), self.output_states[0].shape, self.output_states[1].shape, self.output_states[2].shape))#(?, 122, lstm_size_each_layer[-1])
                #self.print_log('outputs:{}, {}'.format(self.outputs.shape, self.output_states.shape))#(?, 122, lstm_size_each_layer[-1])
        
        #attention(对output做attention的处理)
        #这儿是不是要取最后面的输出来计算(outputs + output_states) tf.transpose([2,1,0])交换纬度第一和第三维度
        #是对一个batch中的每句话取权重这个对么?是不是该对每句话中的单词做权重处理,这样就是关心每句话中的关键的字??
        takeall = False
        if 'True' == takeall:
            inputsize_batch = self.args.max_document_lenth * int(lstm_size_each_layer[-1])
        else:
            inputsize_batch = int(lstm_size_each_layer[-1])
        if self.get_option('summary', 'use_attention', 'int'):
            with tf.name_scope('attention'), tf.variable_scope('attention'):
                attention_size = self.get_option('summary', 'attention_size', 'int')
                attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
                u_list = []
                if use_bidirectional:
                    for index in range(2):
                        attention_w = tf.Variable(tf.truncated_normal([inputsize_batch, attention_size], stddev=0.1), name='attention_w')
                        if 'True' == takeall:
                            self.outputs_flat = tf.reshape(self.outputs[index], [-1, inputsize_batch])
                        else:
                            #tf.transpose(self.outputs[index], [1, 0, 2]) 是将dim(batch_size,steps,inputsize)==>dim(steps,batch_size,inputsize)
                            #tf.unstack(tf.transpose(self.outputs[index], [1, 0, 2]))[-1]取output中的最后一步,本来就是上一个状态和输出会作为下一个的输入,所以直接取最后一笔是OK的
                            self.outputs_flat = tf.unstack(tf.transpose(self.outputs[index], [1, 0, 2]))[-1]
                        u_list.append(tf.tanh(tf.matmul(self.outputs_flat, attention_w) + attention_b))
                else:
                    attention_w = tf.Variable(tf.truncated_normal([inputsize_batch, attention_size], stddev=0.1), name='attention_w')
                    if 'True' == takeall:
                        self.outputs_flat = tf.reshape(self.outputs, [-1, inputsize_batch])
                    else:
                        self.outputs_flat = tf.unstack(tf.transpose(self.outputs, [1, 0, 2]))[-1]
                    u_list.append(tf.tanh(tf.matmul(self.outputs_flat, attention_w) + attention_b))#(?, 122, attention_size)
                
                attn_z = []
                u_w = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1), name='attention_uw')
                for index in range(len(u_list)):
                    z_t = tf.matmul(u_list[index], u_w)
                    attn_z.append(z_t)
                    self.print_log('z_t:{}'.format(z_t.shape))
                
                attn_zconcat = tf.concat(attn_z, axis=1)
                self.print_log('attn_zconcat:{}'.format(attn_zconcat.shape))
                self.alpha = tf.nn.softmax(attn_zconcat)
                if use_bidirectional:
                    self.alpha = tf.reshape(self.alpha, [-1, 2])
                    self.print_log('self.alpha:{},{},{}'.format(self.alpha.shape, self.alpha[:, 0].shape, self.alpha[:, 1].shape))
                
                final_output_tmp = []
                for index in range(len(u_list)):
                    if use_bidirectional:
                        if 'True' == takeall:
                            self.outputs_flat = tf.reshape(self.outputs[index], [-1, inputsize_batch])
                        else:
                            self.outputs_flat = tf.unstack(tf.transpose(self.outputs[index], [1, 0, 2]))[-1]
                        final_output_tmp.append(self.outputs_flat * (tf.reshape(self.alpha[:, index], [-1, 1])))#
                    else:
                        if 'True' == takeall:
                            self.outputs_flat = tf.reshape(self.outputs, [-1, inputsize_batch])
                        else:
                            self.outputs_flat = tf.unstack(tf.transpose(self.outputs, [1, 0, 2]))[-1]
                        self.final_output = self.outputs_flat * self.alpha
                if use_bidirectional:
                    self.final_output = tf.concat(final_output_tmp, axis=1)
                self.print_log('final_output:{}'.format(self.final_output.shape))#3904*512
        else:
            final_output_tmp = []
            if use_bidirectional:
                for index in range(2):
                    if 'True' == takeall:
                        final_output_tmp.append(tf.reshape(self.outputs[index], [-1, inputsize_batch]))#
                    else:
                        final_output_tmp.append(tf.unstack(tf.transpose(self.outputs[index], [1, 0, 2]))[-1])#
            else:
                if 'True' == takeall:
                    self.final_output = tf.reshape(self.outputs, [-1, inputsize_batch])
                else:
                    self.final_output = tf.unstack(tf.transpose(self.outputs, [1, 0, 2]))[-1]
            if use_bidirectional:
                self.final_output = tf.concat(final_output_tmp, axis=1)

        #full connection layer
        with tf.name_scope("output"):
            real_size = inputsize_batch
            if use_bidirectional:
                real_size *= 2
            fc_w = tf.Variable(tf.truncated_normal([real_size, self.args.num_class], stddev=0.1), name='fc_w')
            fc_b = tf.Variable(tf.zeros([self.args.num_class]), name='fc_b')
            self.logits = tf.nn.xw_plus_b(self.final_output, fc_w, fc_b, name="logits")
            self.scores = tf.nn.softmax(self.logits)#每个文本的问题是某类的得分
            self.predictions = tf.argmax(self.logits, 1, name="predictions")#最大得分的index
            self.print_log('scores&predictions:{},{}'.format(self.scores.shape, self.predictions))
        
        #loss
        with tf.name_scope("loss"):
            if 5 <= int(re.findall('\.(.*)?\.', tf.__version__)[0]):
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            else:
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)
        
        with tf.name_scope("accuracy"):
            #self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.input_y, axis=1), self.predictions), tf.float32))
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))#准确率
            self.acc = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
        
        #Optimizer
        with tf.name_scope("Optimizer"):
            tvars = tf.trainable_variables()
            #gradient clippling的方式来防止梯度爆炸,当gradients超过这个阈值时,就将它重置为阈值大小,这就保证了梯度不会变得很大
            #tf.gradients(loss, tvars)loss对所有可训练的梯度
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),self.get_option('summary', 'grad_clip', 'float'))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.get_option('summary', 'learning_rate', 'float'))
            self.optim = optimizer.apply_gradients(zip(grads, tvars))#更新梯度