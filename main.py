"""
This file is used for the main function of the program
which is face recognition, to identify whether the person belongs to the group
Data are set as follows:
  Each person, contains several images of its faces, and is preprocessed to face-descriptors (128D)
  Data are stored in a .cvs file, and will be loaded when program initialize.

* new person: see create_new.py
"""

import cv2
import numpy as np
import pandas as pd
import os
import ast


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

        print('------------------------------init complete------------------------------')
        pass

    def _release_resources(self):
        self.camera.release()
        cv2.destroyAllWindows()
        print('----------------------------resources released----------------------------')

    def convert2list(self):
        self.person_info['descriptors'] = self.person_info['descriptors'].apply(lambda x: ast.literal_eval(x))

    """
    Using web cam to capture faces and recognize, Including two parts: capture & recognize
    each part is processed in one thread, 2 threads total
    """
    def video_capture(self):
        print("Status: Capture Begin!")
        while True:
            _, frame = self.camera.read()
            # TODO: use another thread to process img recognition

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