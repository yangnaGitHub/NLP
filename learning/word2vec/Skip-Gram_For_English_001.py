# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 16:44:58 2018

@author: natasha1_Yang
"""

import time
import numpy as np
import tensorflow as tf
import random
from collections import Counter

def loading_data():
    with open('Skip-Gram_For_English_001', 'r') as fd:
        words = fd.read()
    return words

def preprocess(text, freq=5):
    #预处理
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <QUESTION_MARK> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()
    words_counts = Counter(words)
    trimmed_words = [word for word in words if words_counts[word] > freq]
    return trimmed_words

words = preprocess(loading_data())
vocab = set(words)
vocab_to_int = {w: c for c, w in enumerate(vocab)}
int_to_vocab = {c: w for c, w in enumerate(vocab)}
int_words = [vocab_to_int[w] for w in words]

#sampling
t = 1e-5
threshold = 0.8
int_word_counts = Counter(int_words)
total_count = len(int_words)
word_freqs = {w: c/total_count for w, c in int_word_counts.items()}
prob_drop = {w: 1 - np.sqrt(t/word_freqs[w]) for w in int_word_counts}
train_words = [w for w in int_words if prob_drop[w] < threshold]

def get_targets(words, idx, window_size=5):
    target_window = np.random.randint(1, window_size+1)
    start_point = idx - target_window if (idx - target_window) > 0 else 0#上限溢界处理
    end_point = idx + target_window
    targets = set(words[start_point: idx] + words[idx+1: end_point+1])#滑窗
    return list(targets)

def get_batchs(words, batch_size, window_size=5):
    n_batches = len(words) // batch_size
    words = words[:n_batches*batch_size]
    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx: idx+batch_size]
        for index, val in enumerate(batch):
            batch_y = get_targets(batch, index, window_size)
            x.extend([val]*len(batch_y))
            y.extend(batch_y)
        yield x, y#生成器节省内存,所有使用range的地方可替换成xrange(iterator),版本不支援所以没有用
    
vocab_size = len(int_to_vocab)
embedding_size = 200
n_sampled = 100
valid_windows = 100
valid_size = 16
epochs = 10
batch_size = 1000
window_size = 10
iteration = 1
realloss = 0
top_k = 8
train_graph = tf.Graph()
with train_graph.as_default():
    inputs = tf.placeholder(tf.int32, shape=[None], name='inputs')
    labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
    embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs)
    softmax_w = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1))
    softmax_b = tf.Variable(tf.zeros(vocab_size))
    loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, labels, embed, n_sampled, vocab_size)
    cost= tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    valid_examples = np.array(random.sample(range(valid_windows), valid_size//2))
    valid_examples = np.append(valid_examples, random.sample(range(1000+valid_windows), valid_size//2))
    valid_size = len(valid_examples)
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True))
    normalized_embedding = embedding / norm
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))
    
    saver = tf.train.Saver()

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1, epochs+1):
        batches = get_batchs(train_words, batch_size, window_size)
        start = time.time()
        for x, y in batches:
            feed = {inputs: x,
                    labels: np.array(y)[:, None]
                    }
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)
    
            realloss += train_loss
            if iteration % 100 == 0:
                end = time.time()
                print("Epoch {}/{}".format(epoch, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(realloss/100),
                      "{:.4f} sec/batch".format((end-start)/100))
                realloss = 0
                start = time.time()
            if iteration % 1000 == 0:
                sim = similarity.eval()#等于sess.run(similarity)
                print(type(sim))
                for index in range(valid_size):
                    valid_word = int_to_vocab[valid_examples[index]]
                    print('index:', sim[index, :])
                    nearest = (-sim[index, :]).argsort()[1: top_k+1]#取负号最大变成最小,然后得到从小到大的index
                    log = 'Nearest to [%s]:' % valid_word
                    for near in range(top_k):
                        close_word = int_to_vocab[nearest[near]]
                        log = '%s %s,' % (log, close_word)
                    print(log)
            iteration += 1
    save_path = saver.save(sess, 'E:\\MyOwner\\MyCode\\python\\SL\\English_001.ckpt')#如果使用相对路径出错,请试一下使用绝对路径,windows使用\\
    embed_mat = sess.run(normalized_embedding)

#%matplotlib inline#内嵌画图,有了%matplotlib inline 就可以省掉plt.show()了(JupterNoteBook使用)
#%config InlineBackend.figure_format = 'retina'#高分辨率显示器适用,呈现分辨率较高的图像
import matplotlib.pyplot as plt
#数据是均匀采样于一个高维欧氏空间中的低维流形,流形学习就是从高维采样数据中恢复低维流形结构
#高维空间中的低维流形,一遍聚类一遍降维,其实也是一种比较好的自动聚类方法
from sklearn.manifold import TSNE#manifold流形学习方法

viz_words = 500
tsne = TSNE()#降维可视化
embed_tsne = tsne.fit_transform(embed_mat[:viz_words, :])#前500个words

fig, ax = plt.subplots(figsize=(14, 14))
for idx in range(viz_words):
    plt.scatter(*embed_tsne[idx, :], color='steelblue')
    plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)#添加解释
