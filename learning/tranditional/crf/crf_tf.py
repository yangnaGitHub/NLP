# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 10:50:19 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def get_batch():
    xs = np.array([[2, 3, 4, 5, 5, 5, 1, 5, 3, 2, 5, 5, 5, 3, 5]])
    res = np.array([[0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1]])
    #xs.shape(1, 15) => xs[:, :, np.newaxis].shape(1, 15, 1) => 新增加一个维度
    return [xs[:, :, np.newaxis], res]

class crf():
    def __init__(self, batch_size, num_words, num_word_vector, num_tags, learning_rate):
        self.batch_size = batch_size
        self.num_words = num_words
        self.num_word_vector = num_word_vector
        self.num_tags = num_tags
        self.learning_rate = learning_rate
        self.__construct()
    
    def __construct(self):
        #?, 122, 128 ==> batch_size, 一次有多少个字, 每个字的是多少维度的向量
        self.input = tf.placeholder(tf.float32, [None, self.num_words, self.num_word_vector], name='input')
        self.label = tf.placeholder(tf.int32, [None, self.num_words], name='label')
        
        self.weights = tf.get_variable('weights', [self.num_word_vector, self.num_tags])
        self.output = tf.reshape(tf.matmul(tf.reshape(self.input, [-1, self.num_word_vector]), self.weights), [self.batch_size, self.num_words, self.num_tags])  
        
        num_words_array = np.full(self.batch_size, self.num_words, dtype=np.int32)
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(self.output, self.label, num_words_array)
        self.pred, viterbi_score = tf.contrib.crf.crf_decode(self.output, transition_params, num_words_array)
        self.cost = tf.reduce_mean(-log_likelihood)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
    
if __name__ == '__main__':
    with tf.Graph().as_default() as graph:
        model = crf(1, 15, 1, 2, 0.05)
        with tf.Session(graph=graph).as_default() as sess:
            sess.run(tf.global_variables_initializer())
            batch = get_batch()
            feed_dict = {
                    model.input:batch[0],
                    model.label:batch[1]
                    }
            
            plt.ion()
            plt.show()
            batch_x = range(len(batch[0].reshape(-1,1)))
            batch_y = batch[1].reshape(-1, 1)
            for index in range(150):
                _, cost, pred = sess.run([model.train_op, model.cost, model.pred], feed_dict=feed_dict)

                batch_yo = pred.reshape(-1, 1)
                plt.clf()
                plt.plot(batch_x, batch_y, 'r', batch_x, batch_yo, 'b--')
                plt.ylim((-1.2, 1.2))
                plt.draw()
                plt.pause(0.3)
                if index % 20 == 0:
                    print('cost: ', round(cost, 4))