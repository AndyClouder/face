"""
Created on Thu Jan 10 16:24:28 2019

@author: DiXuanA
"""

import cv2
import os
import dlib
from skimage import io
import csv
import numpy as np
import pandas as pd
import pickle as pk
import eduai_helper as helper
import eduai_api as api

path_faces_rd = "data/cluster_output/"
path_csv = "data/csvs_from_video/"
path_csv_feature_all = "data/features_all.csv"
helper.reset_folder(path_csv)

detector = api.get_face_detector()
facerec = api.get_face_recognizer()
predictor = api.get_68_point_predictor()

def return_128d_features(path_img):
    img = io.imread(path_img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector(img_gray, 1)

    print("face_image: ", path_img, "\n")

    if len(faces) != 0:
        shape = predictor(img_gray, faces[0])
        face_descriptor = facerec.compute_face_descriptor(img_gray, shape)
    else:
        face_descriptor = 0

    return face_descriptor


def write_into_csv(path_faces_personX, path_csv):
    dir_pics = os.listdir(path_faces_personX)
    with open(path_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(dir_pics)):
            print("read the face image: ", path_faces_personX + "/" + dir_pics[i])
            features_128d = return_128d_features(path_faces_personX + "/" + dir_pics[i])
            print(type(features_128d))
            if features_128d == 0:
                i += 1
            else:
                writer.writerow(features_128d)

faces = os.listdir(path_faces_rd)
for person in faces:
    print("This" + person + " face will be saved!")
    print(path_csv + person + ".csv")
    write_into_csv(path_faces_rd + person, path_csv + person + ".csv")



def compute_the_mean(path_csv):
    column_names = []

    # 128
    for feature_num in range(128):
        column_names.append("features_" + str(feature_num + 1))

    rd = pd.read_csv(path_csv, names=column_names)

    feature_mean = []

    for feature_num in range(128):
        tmp_arr = rd["features_" + str(feature_num + 1)]
        tmp_arr = np.array(tmp_arr)

        tmp_mean = np.mean(tmp_arr)
        feature_mean.append(tmp_mean)
    return feature_mean

names_map = {}


with open(path_csv_feature_all, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    csv_rd = os.listdir(path_csv)
    print("The average features stored in: ")

    for i,item in enumerate(csv_rd):
        feature_mean = compute_the_mean(path_csv + csv_rd[i])
        # print(feature_mean)
        writer.writerow(feature_mean)
        names_map[i] = item.replace('.csv','')
    print(path_csv_feature_all)

with open('names_map.pk','wb') as file:
    pk.dump(names_map,file)
