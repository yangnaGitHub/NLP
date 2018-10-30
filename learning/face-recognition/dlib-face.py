# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 14:07:04 2018

@author: natasha_yang

@e-mail: ityangna0402@163.com
"""

#pip install dlib
#pip install scikit-image
#shape_predictor_68_face_landmarks.dat
#dlib_face_recognition_resnet_model_v1.dat
#http://dlib.net/files/

import sys
import os
import dlib
import glob
import numpy

from skimage import io

#人脸关键点检测器 ==> shape_predictor_68_face_landmarks.dat
predictor_path = sys.path[1]
#人脸识别模型 ==> dlib_face_recognition_resnet_model_v1.dat
face_rec_model_path = sys.path[2]
#候选人脸文件夹
faces_folder_path = sys.path[3]
#需识别的人脸
img_path = sys.path[4]

#加载正脸检测器
detector = dlib.get_frontal_face_detector()
#加载人脸关键点检测器
sp = dlib.shape_predictor(predictor_path)
#加载人脸识别模型
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

#候选人脸描述子list
descriptors = []
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)
    
    #人脸检测
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        #关键点检测
        shape = sp(img, d)
        #描述子提取，128D向量
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        v = numpy.array(face_descriptor)
        descriptors.append(v)

#当前要判断的和候选人之间的欧式距离列表
dist = []
#对需识别人脸进行同样处理
img = io.imread(img_path)
dets = detector(img, 1)
for k, d in enumerate(dets):
    shape = sp(img, d)
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    d_test = numpy.array(face_descriptor)
    #计算欧式距离
    for i in descriptors:
        dist_ = numpy.linalg.norm(i-d_test)
        dist.append(dist_)

#候选人名单
candidate = ['Unknown1','Unknown2','Unknown3','Unknown4','Unknown5','Unknown6']

c_d = dict(zip(candidate, dist))
cd_sorted = sorted(c_d.iteritems(), key=lambda d:d[1])#按照欧式距离排序
print("\n The person is: ", cd_sorted[0][0])

#总结
#就是首先要有人脸关键点检测器 + 人脸识别模型
#然后计算待测试人描述子的和每一个候选人描述子的欧式距离最小的