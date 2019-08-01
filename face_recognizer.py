"""
Created on Tue Jan 29 17:46:12 2019

@author: DiXuanA
"""
import numpy as np
import cv2
import pandas as pd
import pickle as pk
import eduai_api as api
import eduai_helper as helper
from eduai_helper import VideoHandler
from logger import Logger
import os
import subprocess

facerec = api.get_face_recognizer()
with open('names_map.pk','rb') as file:
    names_map = pk.load(file)

video = 'TestVideo/test1.mp4'
if os.path.exists(video) == False:
    print("{0} not found!".format(video))


path_features_known_csv = "data/features_all.csv"
csv_rd = pd.read_csv(path_features_known_csv, header=None)

features_known_arr = []

for i in range(csv_rd.shape[0]):
    features_someone_arr = []
    for j in range(0, len(csv_rd.ix[i, :])):
        features_someone_arr.append(csv_rd.ix[i, :][j])
    features_known_arr.append(features_someone_arr)

detector = api.get_face_detector()
predictor = api.get_68_point_predictor()

logger = Logger('eduai.log')
logger.close()
logger.info("############## Start ###############")
vh = VideoHandler(video)
sendmail_flag = 0
frame_number = 0

temp = 0

while vh.vc.isOpened():
    sendmail_flag = 0
    frame_number +=1
    s, frame = vh.vc.read()
    small_frame, faces = vh.get_scale_faces(frame,0.6)
    kk = cv2.waitKey(1)

    pos_namelist = []
    name_namelist = []

    if kk == ord('e'):
        logger.info("############## End ###############")
        exit(1)
        break
    else:
        if len(faces) != 0:
            features_cap_arr = []
            for i in range(len(faces)):
                shape = predictor(small_frame, faces[i])
                features_cap_arr.append(facerec.compute_face_descriptor(small_frame, shape))

            for k in range(len(faces)):
                name_namelist.append("unknown")
                pos_namelist.append(tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                
                left = faces[k].left()
                top = faces[k].top()
                right = faces[k].right()
                bottom = faces[k].bottom()
                height = (bottom - top)
                width = (right - left)
                hh = int(height/3)
                ww = int(width/3)
                
                distances = np.array([])
                for i in range(len(features_known_arr)):
                    print("with person_", str(i+1), "the ", end='')                  
                    distances = np.append(distances,api.return_euclidean_distance(features_cap_arr[k], features_known_arr[i]))
                if distances.min() > 0.5:
                    logger.info(distances.argmin())
                    name_namelist[k] = 'Unknown'
                    logger.critical("unkown person comming!!! It face has been saved to unknown_face_{0}".format(str(frame_number)))
                    face_image = small_frame[top-hh:bottom+hh,left-ww:right+ww]
                    cv2.imwrite("data/unknown_faces" + "/unknown_face_" + str(frame_number) + ".jpg", face_image)
                    #message_unknownFace='UNKNOWN FACE FOUND!'
                    subprocess.call("face_test.sh", shell=True)

                    dis = api.return_euclidean_distance(temp, features_cap_arr[k])
                    if dis>0.51:
                        temp = features_cap_arr[k]
                        logger.warning("new unknown face has been founded!")
                        file = r'\data\unknown_faces\unknown_face_{0}.jpg'.format(frame_number)
                        #logger.sendemail("Warining! ","Unknown people comming in",file)
                else:
                    name_namelist[k] = names_map[distances.argmin()]
                    logger.info(name_namelist[k])
                for face in faces:
                    vh.bounding_box(small_frame,(face.top(),face.bottom(),face.left(),face.right()),c='white')
                    
            for i in range(len(faces)):
                vh.video_print(small_frame,name_namelist[i],location = pos_namelist[i],c='white')
                
    print("Name list now:", name_namelist, "\n")

    vh.video_print(small_frame,"Face recognizer",location = (20,40),c='green')

    cv2.imshow("EDU.AI", small_frame)

vh.vc.release()
cv2.destroyAllWindows()
