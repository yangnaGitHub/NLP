# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 11:57:55 2019

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

#https://github.com/ageitgey/face_recognition
#https://github.com/ageitgey/face_recognition#installation-options
#https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf
import os
import face_recognition

image_tb_matched = face_recognition.load_image_file('compare.jpg')
image_tb_matched_encoded = face_recognition.face_encoding(image_tb_matched)[0]#人物脸部图片转化为特征向量

allimages = os.listdir('allimages')
for image in allimages:#这是遍历所有的待比较的图片
    current = face_recognition.load_image_file('allimages/' + image)
    current_encoded = face_recognition.face_encoding(current)[0]
    result = face_recognition.compare_faces([image_tb_matched_encoded], current_encoded)
    if result[0]:
        print('matched: ', image)
    else:
        print('not matched: ', image)