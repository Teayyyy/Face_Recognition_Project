"""
This file is used for the main function of the program
which is face recognition, to identify whether the person belongs to the group
Data are set as follows:
  Each person, contains several images of its faces, and is preprocessed to face-descriptors (128D)
  Data are stored in a .cvs file, and will be loaded when program initialize.

* new person: see create_new.py
"""

import cv2
import dlib
import numpy as np
import pandas as pd
import ast
import threading


class Face_Recognition:
    """
    Initialization: Load all exist people, data will be stored in test_faces_inner_outer_moments.csv
        Caution: In column called 'descriptors', it's stored in String, should use ast.literal_eval() to convert
                also, try to use description like: self_face = random.sample(self_faces, 1)[0] <--- 0 matters!!
    """
    def __init__(self):
        self.person_info = pd.read_csv('test_faces_inner_outer_moments.csv')
        assert len(self.person_info) > 0, "Must over 1 person!"
        self.convert2list()
        # print(type(self.person_info['descriptors'][0]))
        print('Data already loaded', len(self.person_info), 'people')
        # Init camera
        self.camera = cv2.VideoCapture(0)
        # Init dlib
        self._face_model = "/Users/outianyi/Computer_Vision/dlib_face_recognition_resnet_model_v1.dat"
        self._shape_model = "/Users/outianyi/Computer_Vision/shape_predictor_68_face_landmarks.dat"
        self._face_recognizer = dlib.face_recognition_model_v1(self._face_model)
        self._shape_predictor = dlib.shape_predictor(self._shape_model)
        self._face_detector = dlib.get_frontal_face_detector()
        # Init rectangles locate person
        self._det_face_info = []
        # Init threads
        self._thread_capture = None
        self._thread_recognize = None
        # Init img
        self._img = None

        print('------------------------------init complete------------------------------')

    def _release_resources(self):
        self.camera.release()
        cv2.destroyAllWindows()
        print('----------------------------resources released----------------------------')

    """
    convert str to list 
    """
    def convert2list(self):
        self.person_info['descriptors'] = self.person_info['descriptors'].apply(lambda x: ast.literal_eval(x))

    """
    calculating distance between two descriptors
    """
    def _calc_distance(self, des_a, des_b):
        return np.sqrt(np.sum(np.square(np.array(des_a) - np.array(des_b))))

    """
    Processing face recognition and return: [face_position, descriptor]
    Mention: only return one face, which is the first face in detected faces
    """
    def _image_processing(self):
        # compute face and descriptors
        if self._img is None:
            return
        det_faces = self._face_detector(self._img, 1)
        if len(det_faces) > 0:
            print('detected!')
            det_face = det_faces[0]
            shape = self._shape_predictor(self._img, det_face)
            descriptor = self._face_recognizer.compute_face_descriptor(self._img, shape)
            # making rectangle of detected face
            left, top, right, bottom = det_face.left(), det_face.top(), det_face.right(), det_face.bottom()
            self._det_face_info = [[left, top, right, bottom], descriptor]
        self._img = None
        # return [[left, top, right, bottom], descriptor]
        # TODO: recognize face in here?

    """
    Using web cam to capture faces and recognize, Including two parts: capture & recognize
    each part is processed in one thread, 2 threads total
    """
    def video_capture(self):
        print("Status: Capture Begin!")
        count: int = 0
        while True:
            # count += 1
            _, frame = self.camera.read()
            # TODO: use another thread to process img recognition
            # if count == 2:
            self._img = frame
            self._image_processing()
            # count = 0
            # use det_face_info to show faces pos on frame
            # TODO: recognize_face if this is familiar person
            if len(self._det_face_info) != 0:
                pos1, pos2 = (self._det_face_info[0][0], self._det_face_info[0][1]), \
                    (self._det_face_info[0][2], self._det_face_info[0][3])
                cv2.rectangle(frame, pos1, pos2, [255, 0, 0], 2)
                pass
            cv2.imshow('frame', frame)

            # exit port
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # release
        self._release_resources()
        pass






def main():
    fr = Face_Recognition()
    fr.video_capture()
    pass


# ------------------------------------------------------------
main()