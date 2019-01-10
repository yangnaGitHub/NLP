# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:13:59 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import tensorflow as tf
import os
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3'

class TextCNN(object):
    def __init__(self, args, params, log=None, opmodule=None):
        os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
        self.args = args
        self.params = params
        self.log = log
        self.opmodule = opmodule
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
    
    def get_shape(self, var):
        return tf.shape(var)

    def construct(self):
        tf.set_random_seed(66)
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, self.args.max_document_lenth], name="input_x")#这是要一次训练None * self.args.max_document_lenth
        self.input_y = tf.placeholder(tf.float32, [None, self.args.num_class], name="input_y")#所有的分类,要不要多加一类??都没有命中(get_labels中确定class的数值)
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")#正则化

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):#词向量#tf.device('/cpu:0')
            if 0 == self.get_option('embedding', 'use_embedding_tf', 'int'):
                #self.embedded_module.o_input_x = self.input_x
                #self.embedded_w = tf.placeholder(tf.float32, [self.args.vocab_size, int(self.get_params('embedding', 'embedding_size'))], name='embedded_w')
                #self.embedded_chars = tf.nn.embedding_lookup(self.embedded_w, self.input_x)#self.embedded_module.o_embedding
                if hasattr(self.opmodule, 'embedded_w'):
                    self.embedded_w = tf.Variable(self.opmodule.get_embedded_w, name='W')
                    self.embedded_chars = tf.nn.embedding_lookup(self.embedded_w, self.input_x)#self.embedded_module.o_embedding
                else:
                    self.print_log('lack embedded_w')
            else:
                self.W = tf.Variable(
                        tf.random_uniform([self.args.vocab_size, self.get_option('embedding', 'embedding_size', 'int')], -1.0, 1.0),#vocab_size(build_vocab_size 最大5000字) * embedding_size [-1, 1]
                        name="W")
                #self.update =tf.assign(self.W,words_embedding)
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)#这个的shape
            
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)#(?, 122, 128)==>(?, 122, 128, 1)cnn需要的一步,多了一维度
            self.print_log('embedding:{}, {}'.format(self.embedded_chars.shape, self.embedded_chars_expanded.shape))#natasha here

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        conv_maxpool_count = self.get_option('summary', 'conv_maxpool_count', 'int')
        for index in range(conv_maxpool_count):
            layername = 'conv-maxpool-%s' % index
            with tf.name_scope(layername):
                # Convolution Layer
                #卷积核 高度 宽度 通道数 卷积核个数
                filter_size = self.get_option(layername, 'filter_size', 'int')
                num_filters = self.get_option(layername, 'num_filters', 'int')
                filter_shape = [filter_size, self.get_option('embedding', 'embedding_size', 'int'), 1, num_filters]#[1, 2, 3, 4, 5] 128 1 128
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")#128
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],#步长
                    padding="VALID",#SAME(with zero padding) VALID(without padding)
                    name="conv")#use_cudnn_on_gpu:bool类型,是否使用cudnn加速,默认为true
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                self.print_log('h:{}, {}'.format(layername, h.shape))#natasha here
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.args.max_document_lenth - filter_size + 1, 1, 1],#self.args.max_document_lenth - filter_size + 1保证每个卷积单元pool出来的维度是一样的
                    strides=[1, 1, 1, 1],#步长,如果步长变了,那么上面每个单元还要保持一致的话就要修改ksize了
                    padding='VALID',
                    name="pool")
                self.print_log('pooled:{}, {}'.format(layername, pooled.shape))#natasha here
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * conv_maxpool_count#128 * len([1, 2, 3, 4, 5])
        self.h_pool = tf.concat(pooled_outputs, 3)#concat成一个
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        self.print_log('combine:{}, {}'.format(self.h_pool, self.h_pool_flat.shape))#natasha here
            
        use_attention = self.get_option('summary', 'use_attention', 'int')
        if 1 == use_attention:
            with tf.name_scope('attention'), tf.variable_scope('attention'):
                attention_w = tf.Variable(tf.truncated_normal([num_filters_total, num_filters_total], stddev=0.1), name='attention_w')
                attention_b = tf.Variable(tf.constant(0.1, shape=[num_filters_total]), name='attention_b')
                u_t = tf.tanh(tf.matmul(self.h_pool_flat, attention_w) + attention_b)
                self.alpha = tf.nn.softmax(u_t)
                self.final_output = self.h_pool_flat * self.alpha
        
        if 2 == use_attention:
            #定义attention layer 
            #attention_size = num_filters_total
            attention_size = self.get_option('summary', 'attention_size', 'int')
            with tf.name_scope('attention'), tf.variable_scope('attention'):
                attention_w = tf.Variable(tf.truncated_normal([num_filters_total, attention_size], stddev=0.1), name='attention_w')
                attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
                #取出的outputs[t]是每个句子输入后得到的结果，[batch_size,2*rnn_size(双向)]
                #因此u_t可以认为是对这个单词结果的打分,[batch_size,attention_size]
                u_t = tf.tanh(tf.matmul(self.h_pool_flat, attention_w) + attention_b)#x * num_filters_total
                #最终得到的u_list是[sequence_length,batch_size,attention_size],是每一个单词的权重
                u_w = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1), name='attention_uw')
                #将权重变为[batch_size,1]
                z_t = tf.matmul(u_t, u_w)#x * 1
                #取概率权重
                self.print_log('z_t:{}'.format(z_t.shape))#natasha here
                self.alpha = tf.nn.softmax(z_t)
                #[batch_size,num_filters_total]*[batch_size,1]=[batch_size,num_filters_total],实际就是对每一个乘以一个权重
                self.final_output = self.h_pool_flat * self.alpha#x * num_filters_total
        
        # Add dropout
        with tf.name_scope("dropout"):
            if 1 == use_attention:
                self.h_drop = tf.nn.dropout(self.final_output, self.keep_prob)#没有加attention layer(没有改变维度)是不是该用self.h_pool_flat
            elif 0 == use_attention:
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_prob)
            else:
                self.print_log('{}'.format('use_attention params error'))#natasha here

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            #使用tf.Variable时,如果检测到命名冲突,系统会自己处理.使用tf.get_variable()时,系统不会处理冲突,而会报错
            #需要共享变量的时候,需要使用tf.get_variable()
            #tf.Variable()每次都在创建新对象,所有reuse=True和它并没有什么关系.对于get_variable()如果已经创建的变量对象,就把那个对象返回,如果没有创建变量对象的话,就创建一个新的.
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.args.num_class],
                initializer=tf.contrib.layers.xavier_initializer())#xavier_initializer一种经典的权值矩阵的初始化方式,用来保持每一层的梯度大小都差不多相同
            b = tf.Variable(tf.constant(0.1, shape=[self.args.num_class]), name="b")
            l2_loss += tf.nn.l2_loss(W)#利用L2范数来计算张量的误差值sum(t ** 2) / 2
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")#matmul(self.h_drop, W) + b.
            self.scores = tf.nn.softmax(self.logits)#batch*num_class
            #预测的输出就是分类
            self.predictions = tf.argmax(self.logits, 1, name="predictions")#0每一列比较返回最大数的index(个数就是列数), 1每一行比较返回最大数的index(个数就是行数)
       
        # CalculateMean cross-entropy loss
        #模型定义好就要定义loss和优化目标(使得loss最小)以及如何优化
        #学习率
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.get_option('summary', 'l2_reg_lambda', 'float') * l2_loss#l2正则 ==> 网络中加入dropout,后面的线性L2正则

        # Accuracy
        with tf.name_scope("accuracy"):
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))#准确率
            self.acc = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
            
        self.optim = tf.train.AdamOptimizer(learning_rate=self.get_option('summary', 'learning_rate', 'float')).minimize(self.loss)