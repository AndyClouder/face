# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 10:43:25 2019

@author: DiXuanA
"""

import cv2
import dlib
import pylab
import imageio
import skimage
import numpy as np
import sys,os
import shutil
from PIL import Image
import facerec_model

def get_face_detector():
    return dlib.get_frontal_face_detector()

def get_face_recognizer():
    face_recognition_model = facerec_model.face_recognition_model_location()
    return dlib.face_recognition_model_v1(face_recognition_model)

def get_68_point_predictor():
    predictor_68_point_model = facerec_model.pose_predictor_model_location()
    return dlib.shape_predictor(predictor_68_point_model)

def return_euclidean_distance(feature_1, feature_2):
    vec_1 = np.array(feature_1)
    vec_2 = np.array(feature_2)
    vec_2 = vec_2.astype('float')
    dist = np.sqrt(np.sum(np.square(vec_1 - vec_2)))
    return dist

def return_cosine_similariy(feature_1,feature_2):
    vec_1 = np.array(feature_1)
    vec_2 = np.array(feature_2)
    return np.dot(vec_1,vec_2)/(np.linalg.norm(vec_1)*(np.linalg.norm(vec_2)))
