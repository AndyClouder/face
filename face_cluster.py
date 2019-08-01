# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:24:19 2019
This is the face images cluster

@author: DiXuanA
"""

import sys
import os
import dlib
import glob
import cv2
import eduai_api as api
from sklearn.cluster import DBSCAN

current_path = os.getcwd() 

face_folder = current_path + '/data/faces_from_video/' 
output_folder = current_path + '/data/cluster_output/'
print(face_folder)
detector = api.get_face_detector()
face_recognizer  = api.get_face_recognizer()
shape_detector  = api.get_68_point_predictor()

descriptors = []
images = []

for f in glob.glob(os.path.join(face_folder, "*.jpg")):
    
    print('Processing file：{}'.format(f))
    img = cv2.imread(f)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(img2, 1)
    
    for index, face in enumerate(dets):
        shape = shape_detector(img2, face)
        face_descriptor = face_recognizer.compute_face_descriptor(img2, shape)
        descriptors.append(face_descriptor)
        images.append((img2, shape))

#clt = DBSCAN(eps=0.35,min_samples=2,metric = "euclidean")
#labels = clt.fit_predict(descriptors)
#labels = labels+1
#images_number = len(descriptors)  
#print("Number of images : {}".format(images_number))   

labels = dlib.chinese_whispers_clustering(descriptors, 0.4)
print("labels: {}".format(labels))
num_classes = len(set(labels))
print("Number of clusters: {}".format(num_classes))

noise = [i==0 for i in list(labels)].count(True)

print("Number of noise : {}".format(noise))
num_classes = len(set(labels))
print("Number of clusters : {}".format(num_classes))
print("labels: {}".format(labels))
face_dict = {}
for i in range(num_classes):
    face_dict[i] = []
    
for i in range(len(labels)):
    face_dict[labels[i]].append(images[i])
    
for key in face_dict.keys():
    file_dir = os.path.join(output_folder, str(key))
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
        
    for index, (image, shape) in enumerate(face_dict[key]):
        file_path = os.path.join(file_dir, 'face_' + str(index))
        dlib.save_face_chip(image, shape, file_path, size=150, padding=0.25)
