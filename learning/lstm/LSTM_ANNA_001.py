# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 19:08:27 2018

@author: natasha1_Yang
"""

import time
from collections import namedtuple
import numpy as np
import tensorflow as tf
import copy

DEBUG_PRINT=1

with open('anna.txt', 'r') as fd:
    text = fd.read()
vocab = set(text)
vocab_to_int = {c: idx for idx, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encode = np.array([vocab_to_int[c] for c in text], dtype=np.int32)#text编码成index

def get_batches(arr, n_seqs, n_steps):
    batch_size = n_seqs * n_steps
    n_batches = len(arr) // batch_size
    arr = arr[:batch_size*n_batches]
    arr = arr.reshape((n_seqs, -1))
    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n: n+n_steps]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y

batches = get_batches(encode, 10, 50)
if DEBUG_PRINT:
    x, y = next(batches)
    print('x\n', x[:10, :10])
    print('y\n', y[:10, :10])

#construct network
def build_inputs(num_seqs, num_steps):#10 * 50
    inputs = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='inputs')
    labels = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, labels, keep_prob

def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    lstm_list = [drop]
    lstm_list.extend([copy.deepcopy(drop) for _ in range(num_layers-1)])#地址的不对导致的错误
    cell = tf.nn.rnn_cell.MultiRNNCell(lstm_list)
    initial_state = cell.zero_state(batch_size, tf.float32)
    return cell, initial_state

def build_output(lstm_output, in_size, out_size):
    seq_output = tf.concat(lstm_output, axis=1)
    x = tf.reshape(seq_output, [-1, in_size])
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))
    logits = tf.matmul(x, softmax_w) + softmax_b
    out = tf.nn.softmax(logits, name='predictions')
    return out, logits

def build_loss(logits, targets, lstm_size, num_classes):
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
    return loss
#RNN会遇到梯度爆炸（gradients exploding）和梯度弥散（gradients disappearing)的问题
#LSTM解决了梯度弥散的问题,但是gradient仍然可能会爆炸,因此我们采用gradient clippling的方式来防止梯度爆炸
#即通过设置一个阈值,当gradients超过这个阈值时,就将它重置为阈值大小,这就保证了梯度不会变得很大
def build_optimizer(loss, learning, grad_clip):
    tvars = tf.trainable_variables()
    #tf.gradients(loss, tvars)loss对所有可训练的梯度
    #FP&BP后得到每个权重的梯度diff,这时不像通常那样直接使用这些梯度进行权重更新,而是先求所有权重梯度的平方和sumsq_diff
    #如果sumsq_diff > grad_clip,则求缩放因子scale_factor = grad_clip / sumsq_diff,
    #这个scale_factor在(0,1)之间,如果权重梯度的平方和sumsq_diff越大,那缩放因子将越小
    #最后将所有的权重梯度乘以这个缩放因子,这时得到的梯度才是最后的梯度信息
    #tf.clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None)
    #t_list[i] * clip_norm / max(global_norm, clip_norm)
    #global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))所有梯度的平方和
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)#让权重的更新限制在一个合适的范围(阈值)
    train_op = tf.train.AdamOptimizer(learning)
    optimizer = train_op.apply_gradients(zip(grads, tvars))#更新梯度
    return optimizer

class CharRNN:
    def __init__(self, num_classes, batch_size=64, num_steps=50, lstm_size=128,
                 num_layers=2, learning_rate=0.001, grad_clip=5, sampling=False):
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps
        tf.reset_default_graph()
        
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)#100, 100
        
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)#512, 2, 100
        
        x_one_hot = tf.one_hot(self.inputs, num_classes)#函数的作用是将一个值化为一个概率分布的向量 100 * 100 * 83
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state
        
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)

batch_size = 100
num_steps = 100
lstm_size = 512
num_layers = 2
learning_rate = 0.001
keep_prob = 0.5
epochs = 20
save_every_n = 20
model = CharRNN(len(vocab), batch_size=batch_size, num_steps=num_steps, lstm_size=lstm_size, num_layers=num_layers ,learning_rate=learning_rate)
saver = tf.train.Saver(max_to_keep=100)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    counter = 0
    for e in range(epochs):
        new_state = sess.run(model.initial_state)
        loss = 0
        for x, y in get_batches(encode, batch_size, num_steps):
            counter += 1
            start = time.time()
            feed = {model.inputs: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([model.loss,
                                                 model.final_state,
                                                 model.optimizer], feed_dict=feed)
            end = time.time()
            if counter % 100 == 0:
                print('轮数: {}/{}... '.format(e+1, epochs),
                      '训练步数: {}... '.format(counter),
                      '训练误差: {:.4f}... '.format(batch_loss),
                      '{:.4f} sec/batch'.format((end-start)))
            if (counter % save_every_n == 0):
                saver.save(sess, 'E:\\MyOwner\\MyCode\\python\\SL\\checkpoints\\LSTN_ANNA_C{}_S{}.ckpt'.format(counter, lstm_size))
    saver.save(sess, 'E:\\MyOwner\\MyCode\\python\\SL\\checkpoints\\LSTN_ANNA_C{}_S{}.ckpt'.format(counter, lstm_size))

tf.train.get_checkpoint_state('checkpoints')#查看checkpoints

#文本生成
def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)#从数组的形状中删除单维条目,即把shape中为1的维度去掉,想压成1维的话用flatten
    p[np.argsort(p)[:-top_n]] = 0#如果是多维不能这么取?
    p = p / np.sum(p)#归一化
    c = np.random.choice(vocab_size, 1, p = p)[0]#1是选择一个
    return c

def sample(checkpoint, n_samples, lstm_size, vocab_size, prime='The '):
    samples = [c for c in prime]
    model = CharRNN(len(vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
    with tf.Session() as sess:
        for c in prime:
            x = np.zeros((1, 1))
            x[0, 0] = vocab_to_int[c]
            feed = {model.inputs: x,#本来是100*100的,输入是1没有问题?
                    model.keep_prob: 1.,
                    model.initial_state: new_state
                    }
            preds, new_state = sess.run([model.prediction, model.final_state], feed_dict=feed)
            print(preds)
            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])
    return ''.join(samples)

#选用最终的训练参数作为输入进行文本生成
checkpoint = tf.train.latest_checkpoint('checkpoints')
samp = sample(checkpoint, 2000, lstm_size, len(vocab), prime='The')
print(samp)

checkpoint = 'LSTN_ANNA_C200_S512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime="Far")
print(samp)
checkpoint = 'LSTN_ANNA_C1000_S512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime="Far")
print(samp)
checkpoint = 'LSTN_ANNA_C2000_S512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime="Far")
print(samp)
#概率图模型 HMM,MEMM,CRF(标注模型)
#cell的层数一共定义了L层,为什么dropout要进行L+1次呢
#就是因为输入这个地方要进行1次dropout(设置cell的hidden units size=200的话,input embbeding dim=200维度较高,dropout一部分,防止overfitting)