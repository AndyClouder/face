# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 10:45:47 2019

@author: DiXuanA
"""

import imageio
import skimage
import shutil
import os
import cv2
import eduai_api as api

def reset_folder(data_dir):
    if os.path.isdir(data_dir):
        print('have')
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)

def color(item):
    color_map = {'blue':(255,0 , 0),
                  'green':(0,255, 0),
                  'red':(0,0,255),
                  'white':(255,255,255),
                  'black':(0,0,0)}
    return color_map[item]

class VideoHandler(object):
    def __init__(self,video):

        self.vc = cv2.VideoCapture(video)
        self.w = self.vc.get(3)
        self.h = self.vc.get(4)
        self.font = cv2.FONT_HERSHEY_COMPLEX
        self.detector = api.get_face_detector()

    def get_scale_faces(self,frame,scale = 0.6):
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        return small_frame, self.detector(gray, 0)

    def video_print(self,small_frame,message,location = (20,300), c= 'white'):
        cv2.putText(small_frame, message, (location[0], location[1]), self.font, 0.8, color(c), 1, cv2.LINE_AA)

    def bounding_box(self, small_frame, locaiton, c = 'white'):
        cv2.rectangle(small_frame, (locaiton[2], locaiton[0]), (locaiton[3], locaiton[1]), color(c), 2)
