import cv2
import numpy as np
import pandas as pd
import dlib
import os
import concurrent.futures
import random

"""
This file is used for calculate all the people's face descriptors, and compute the moments of each person.
To run this file, you need to:
    * Generate a .xlsx or .csv file contains path to each person's directory, like "test_people_path.csv"
    * Preinstall models of dlib: 
        - dlib_face_recognition_resnet_model_v1.dat
        - shape_predictor_68_face_landmarks.dat
You may need to modify max_workers in multi-threading, which is 6 in default 
"""

# loading models, detectors
facerec_model = "/Users/outianyi/Computer_Vision/dlib_face_recognition_resnet_model_v1.dat"
face_recognizer = dlib.face_recognition_model_v1(facerec_model)
model = '/Users/outianyi/Computer_Vision/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(model)
detector = dlib.get_frontal_face_detector()


# functions
class FaceHelper:
    @staticmethod
    def distance(a, b):
        # return np.linalg.norm(a - b)
        return np.sqrt(np.sum(np.square(np.array(a) - np.array(b))))

    # only calc one person's face each time
    # face_path: the dir contains all faces of one person
    @staticmethod
    def calc_all_vectors(face_path):
        # print('.', end='')
        face_vectors = []
        for filename in os.listdir(face_path):
            t_path = os.path.join(face_path, filename)
            t_img = cv2.imread(t_path, cv2.IMREAD_COLOR)
            det_face = detector(t_img, 1)
            # print(det_face)
            if len(det_face) > 0:
                det_face = det_face[0]
                t_shape = predictor(t_img, det_face)
                t_descriptor = face_recognizer.compute_face_descriptor(t_img, t_shape)
                face_vectors.append([list(t_descriptor)])
        return face_vectors

    # face_vectors: contains all face descriptors under one person
    @staticmethod
    def calc_average_moments(face_vectors):
        moments = []
        if len(face_vectors) > 0:
            for i in range(len(face_vectors)):
                for j in range(i, len(face_vectors)):
                    moments.append(FaceHelper.distance(face_vectors[i], face_vectors[j]))
            return np.mean(moments)
        else:
            return 0

    # choose one vector in face_vectors and calc moments with one in random_vectors
    @staticmethod
    def calc_diff_moments(face_vectors: list, random_vectors: list):
        t_face_vector = random.sample(face_vectors, 1)
        moments = []
        for t_vectors in random_vectors:
            t_vector = random.sample(t_vectors, 1)
            moments.append(FaceHelper.distance(t_face_vector, t_vector))
        return np.mean(moments)
    pass

    # randomly generate 5 vectors
    @staticmethod
    def random_choice_5paths(vectors, batch_size=5):
        vectors_list = list(vectors)
        while True:
            random_vectors = random.sample(vectors_list, batch_size)
            yield random_vectors


# read all paths that contains over 2 faces
# ovr_2faces_list = pd.read_excel('Ovr_2faces_path.xlsx')
# read test faces
ovr_2faces_list = pd.read_csv('test_people_path.csv')
# ovr_2faces_list.rename(columns={'Unnamed: 0': 'index', 0: 'path'}, inplace=True)


# test, calc 200
ovr_2faces_list = ovr_2faces_list[: 400]

def processing_method(ovr_2faces_list):
    # make names
    print('making names...')
    get_name = lambda x: x.split('/')[-1]
    ovr_2faces_list['name'] = ovr_2faces_list['path'].apply(get_name)

    # get all descriptors
    print('making descriptors...')
    ovr_2faces_list['descriptors'] = ovr_2faces_list['path'].apply(FaceHelper.calc_all_vectors)

    # calc all mean moments of one person's face
    print('calculating mean moments of each persons face...')
    ovr_2faces_list['average moments'] = ovr_2faces_list['descriptors'].apply(FaceHelper.calc_average_moments)

    # reorder
    print('reordering...')
    new_index = ['index', 'path', 'name', 'average moments', 'descriptors']
    ovr_2faces_list = ovr_2faces_list.reindex(columns=new_index)

    # # save
    # print('saving...')
    # ovr_2faces_list.to_csv("LFW_ovr2_faces_info.csv")
    return ovr_2faces_list


chunk_size = 200
# divide data into several chunks
chunks = [ovr_2faces_list[i: i + chunk_size] for i in range(0, len(ovr_2faces_list), chunk_size)]

# create thred pool, max cpu cores is 6
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    results = executor.map(processing_method, chunks)

results = list(results)
results = pd.concat(results)

print('saving...')
# results.to_csv('multi_threding_results111.csv')
# calc_test faces
results.to_csv('test_people_results.csv')
