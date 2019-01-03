# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 15:52:16 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

#编解码框架内,编码段/解码段加入attention模型,就是对源数据的加权变换
#可以提高序列的自然方式下的系统表现
#
#空间注意力spatial attention+时间注意力temporal attention
#soft attention:可被嵌入到模型中去进行训练并传播梯度
#hard attention:不计算所有输出,依据概率对encoder的输出采样,在反向传播时需采用蒙特卡洛进行梯度估计
#global attention:对所有encoder输出进行计算
#local attention:介于soft和hard之间,会预测一个位置并选取一个窗口进行计算
#Self Attention:分别计算Q和K自身的依赖关系
#
#1>为何要加入attention
#  序列的不断增长,原始根据时间步的方式表现的越来越差(上下文输入信息被限制到固定长度)
#  
#attention就是另外一个模型,这个模型负责给输入的上下文信息进行打分,解码器的预处理过程,告诉解码器哪些编码器的输出的内容重要
#
#seq2seq:一个encoder和一个decoder,将一个输入的句子

#蕴含关系推理（Entailment Reasoning）
 #Reasoning about Entailment with Neural Attention

#tf相关函数的解析
#https://zhuanlan.zhihu.com/p/27769667
#https://blog.csdn.net/u012436149/article/details/52976413

import tensorflow as tf

#输入(batch_size,max_document_lenth,embedding_size)
 #batch_size个句子,每个句子的长度是max_document_lenth,每个字的向量表示维度是embedding_size
#输出(batch_size,max_document_lenth,p_size)
 #batch_size个句子,每个句子的长度是max_document_lenth,每个字的位置表示的维度是p_size
 #p * (1/10000^(2i/p_size))
def Calc_PE(inputs, p_size):
    batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
    #position
    w_position = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
    #若p_size是基数会多一个
    p_position = 1 / tf.pow(10000., 2 * tf.range(p_size / 2, dtype=tf.float32) / p_size)
    h_position = tf.matmul(tf.expand_dims(w_position, 1), tf.expand_dims(p_position, 0))#seq_len * p_size / 2
    #   1*(1/10000^(0/p_size))        1*(1/10000^(2/p_size))  ...        1*(1/10000^(2*((p_size-1)//2)/p_size))
    #   2*(1/10000^(0/p_size))        2*(1/10000^(2/p_size))  ...        2*(1/10000^(2*((p_size-1)//2)/p_size))
    #       ...                           ...                 ...            ...
    #   seq_len*(1/10000^(0/p_size))  seq_len*(1/10000^(2/p_size))  ...  seq_len*(1/10000^(2*((p_size-1)//2)/p_size))
    
    #1 * seq_len * [2*((p_size-1)//2+1)]
    a_position = tf.expand_dims(tf.concat([tf.cos(h_position), tf.sin(h_position)], 1), 0)#一句话的PE顺序是[2i+1, 2i](i=0~((p_size-1)//2))
    #1 * seq_len * [2*((p_size-1)//2+1)]和2 * seq_len * [2*((p_size-1)//2+1)]是一样的
    return a_position + tf.zeros((batch_size, seq_len, 2*((p_size-1)//2+1)))#扩展成batch_size * seq_len * [2*((p_size-1)//2+1)]

def Mask(outputs, seq_len, mode='mul'):
    if None == seq_len:
        return outputs
    if mode not in ['mul', 'add']:
        return outputs
    mask = tf.cast(tf.sequence_mask(seq_len), tf.float32)
    for index in range(len(outputs.shape)-2):
        mask = tf.expand_dims(mask, 2)
    if 'mul' == mode:
        return outputs * mask
    if 'add' == mode:
        return outputs - (1 - mask) * 1e12

#一次线性操作(也就是全连接层)
#inputs(x, y ,z) * w(z, k) + b(k) = (x, y, k)outputs
def Dense(inputs, output_size, bias=True, seq_len=None):
    input_size = int(inputs.shape[-1])#z
    w = tf.Variable(tf.random_uniform([input_size, output_size], -0.05, 0.05))#(z, k)
    print(tf.shape(w))
    if bias:
        b = tf.Variable(tf.random_uniform([output_size], -0.05, 0.05))
    else:
        b = 0
    outputs = tf.matmul(tf.reshape(inputs, (-1, input_size)), w) + b#(x*y, z)
    outputs = tf.reshape(outputs, tf.concat([tf.shape(inputs)[:-1], [output_size]], 0))#(x, y, z)
    if seq_len != None:
        outputs = Mask(outputs, seq_len, 'mul')
    return outputs
    
def attention(Q, K, V, c_head, size_per_head, Q_len=None, V_len=None):
    #Q(x, y, z) => Q(x, y, c_head*size_per_head)
    Q = Dense(Q, c_head*size_per_head, False)#Q线性变换
    #Q(x, y, c_head*size_per_head) => Q(x, y, c_head, size_per_head)
    Q = tf.reshape(Q, (-1, tf.shape(Q)[1], c_head, size_per_head))
    #QQ(x, y, c_head, size_per_head) => Q(x, c_head, y, size_per_head)
    Q = tf.transpose(Q, [0, 2, 1, 3])
    print(tf.shape(Q))
    
    K = Dense(K, c_head*size_per_head, False)#K线性变换
    K = tf.reshape(K, (-1, tf.shape(K)[1], c_head, size_per_head))
    K = tf.transpose(K, [0, 2, 1, 3])
    
    V = Dense(V, c_head*size_per_head, False)#V线性变换
    V = tf.reshape(V, (-1, tf.shape(V)[1], c_head, size_per_head))
    V = tf.transpose(V, [0, 2, 1, 3])
    
    #没有弄清楚多维矩阵的乘法:1,2,3,4*4,3,2,1=1,2,3,3
    temp = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(size_per_head))
#    temp = tf.transpose(temp, [0, 3, 2, 1])
#    temp = Mask(temp, V_len, mode='add')
#    temp = tf.transpose(temp, [0, 3, 2, 1])
    temp = tf.nn.softmax(temp)
    
    O = tf.matmul(temp, V)
    O = tf.transpose(O, [0, 2, 1, 3])
    O = tf.reshape(O, (-1, tf.shape(O)[1], c_head * size_per_head))
    O = Mask(O, Q_len, 'mul')
    return O

#Q = tf.Variable(tf.random_uniform([1, 2, 3, 4], -1, 1))
#K = tf.Variable(tf.random_uniform([1, 2, 3, 4], -1, 1))
#V = tf.Variable(tf.random_uniform([1, 2, 3, 4], -1, 1))
#temp = tf.matmul(Q, K, transpose_b=True)
#final = tf.global_variables_initializer()
#with tf.Session() as sess:
#    sess.run(init)
#    print(sess.run(tf.shape(tf.transpose(V))))
#    print(sess.run(tf.shape(final)))