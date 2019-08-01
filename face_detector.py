"""
Created on Thu Jan 10 16:18:12 2019

@author: DiXuanA
"""

import cv2
import imageio
import skimage
import numpy as np
import sys,os
import shutil
import eduai_helper as helper
from eduai_helper import VideoHandler
import eduai_api as api

from PIL import Image

detector = api.get_face_detector()

# face counter
max_save_faces_num = 10
person_cnt = 0
# save path
faces_dir = "data/faces_from_video/"

scale = 0.6
video = 'TestVideo/test1.mp4'

if os.path.exists(video) == False:
    print("{0} not found!".format(video))

try:
	if sys.argv[1] is not None:
		video = sys.argv[1]
except IndexError as e:
	print('default video')

mode = {'normal':0,
		'advanced':3}

m = 'normal'

try:
	if sys.argv[2] is not None:
		m = str(sys.argv[2])
except IndexError as e:
	print('normal')


vh = VideoHandler(video)
helper.reset_folder(faces_dir)

frame_nums = 0
face_save_count = 0
start_save = False
while vh.vc.isOpened():
    s, frame = vh.vc.read()
    kk = cv2.waitKey(1)
    small_frame, faces = vh.get_scale_faces(frame,0.6)
    face_count_one_img = 0
    if len(faces) > 0:
        frame_nums += 1
        for face in faces:
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()
            height = (bottom - top)
            width = (right - left)
            hh = int(height/3)
            ww = int(width/3)
            
            face_location = (top-hh,bottom+hh,left-ww,right+ww)
            can_save = True
            color_rectangle = 'white'
            if (right+ww) > vh.w or (bottom+hh > vh.h) or (left-ww < 0) or (top-hh < 0):
                color_rectangle = 'red'
                vh.video_print(small_frame,'out of range',c = color_rectangle)
                can_save = False

            vh.bounding_box(small_frame,face_location,c=color_rectangle)

            if can_save:
                if face_save_count < max_save_faces_num:
                    face_save_count += 1
                    face_count_one_img += 1
                    face_image = small_frame[top-hh:bottom+hh,left-ww:right+ww]
                    cv2.imwrite(faces_dir + "/img_face_" + str(face_count_one_img)+"_"+ str(frame_nums) + ".jpg", face_image)
                    print("save", str(faces_dir) + "/img_face_" + str(face_count_one_img)+"_"+ str(frame_nums) + ".jpg")
                    vh.video_print(small_frame,
                                   "Auto Saving",
                                   c=color_rectangle
                                   )
                    face_count_one_img = 0
                elif start_save == True:
                    face_save_count =0
    else:
        start_save = True
    #print face numbers
    vh.video_print(small_frame,
	               "Face detector:{0}".format(str(len(faces))),
	               c='green',
	               location = (20,50)
                    )
    cv2.imshow("EDU.AI", small_frame)
    if  kk== ord('e'):
        break

vh.vc.release()
cv2.destroyAllWindows()
