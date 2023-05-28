"""
This file is used for the main function of the program
which is face recognition, to identify whether the person belongs to the group
Data are set as follows:
  Each person, contains several images of its faces, and is preprocessed to face-descriptors (128D)
  Data are stored in a .cvs file, and will be loaded when program initialize.

EXIT PROGRAM: Q
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
        self.camera = cv2.VideoCapture(1)
        # Init dlib
        self._face_model = "/Users/outianyi/Computer_Vision/dlib_face_recognition_resnet_model_v1.dat"
        self._shape_model = "/Users/outianyi/Computer_Vision/shape_predictor_68_face_landmarks.dat"
        self._face_recognizer = dlib.face_recognition_model_v1(self._face_model)
        self._shape_predictor = dlib.shape_predictor(self._shape_model)
        self._face_detector = dlib.get_frontal_face_detector()
        # Init threshold for recognition, this can be modified by user
        self._threshold = 0.54
        # Init rectangles locate person
        self._det_face_info = []
        # Init threads
        self._thread_capture = None
        self._thread_recognize = None
        # Init img
        self._img_available = None
        # If person in camera is recognized as familiar person, and name of it
        self._if_recognized = False
        self._recognized_name = 'UNKNOWN'
        # Init thread lock
        self._lock_thread = threading.Lock()
        self._exit_thread = False

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
        print('----------------------- descriptors to list complete -----------------------')

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
        while True:
            if self._exit_thread: break
            # check if any img available
            with self._lock_thread:
                if self._img_available is None:
                    continue

            det_faces = self._face_detector(self._img_available, 1)
            if len(det_faces) > 0:
                det_face = det_faces[0]
                shape = self._shape_predictor(self._img_available, det_face)
                descriptor = self._face_recognizer.compute_face_descriptor(self._img_available, shape)
                # making rectangle of detected face
                left, top, right, bottom = det_face.left(), det_face.top(), det_face.right(), det_face.bottom()
                self._det_face_info = [[left, top, right, bottom], descriptor]
                self._get_familiar_face(descriptor)

            self._img_available = None
            # return [[left, top, right, bottom], descriptor]

            with self._lock_thread:
                self._img_available = None
        # end while

    """
    The method compute all faces of each person, and find min average_distance, 
        then check if average_distance < threshold, if true, recognized as familiar person
    If the face recognized as familiar person, return [True, 'name'], else return [False, '']
    """
    def _get_familiar_face(self, face_descriptor):
        all_calc_distance = []
        for descriptors in self.person_info['descriptors']:
            temp_distances = []
            # calc mean distance in one person
            for descriptor in descriptors:
                # print(len(descriptor[0]), ", ", len(face_descriptor))
                t_dis = self._calc_distance(face_descriptor, descriptor[0])
                temp_distances.append(t_dis)
            all_calc_distance.append(np.mean(temp_distances))
        print("all distance: ", all_calc_distance)

        # get minimum distance, and return its name
        min_val, min_ind = min(all_calc_distance), all_calc_distance.index(min(all_calc_distance))
        print("Min val", min_val)
        if min_val <= self._threshold:
            # return True and name of person info
            self._if_recognized = True
            self._recognized_name = self.person_info['name'][min_ind]
            print('name: ', self._recognized_name)
            # return [True, self.person_info['name'][min_ind]]
        else:
            # return [False, "UNKNOWN"]
            self._if_recognized = False
            self._recognized_name = 'UNKNOWN'

    """
    Using web cam to capture faces and recognize, Including two parts: capture & recognize
    each part is processed in one thread, 2 threads total
    """

    def video_capture(self):
        print("Status: Capture Begin!")
        count: int = 0
        while True:
            count += 1
            _, frame = self.camera.read()
            # use another thread to process img recognition, see start()
            if count == 5:
                self._img_available = frame
                count = 0
            # self._image_processing()
            # use det_face_info to show faces pos on frame
            # TODO: recognize_face if this is familiar person
            if len(self._det_face_info) != 0:
                pos1, pos2 = (self._det_face_info[0][0], self._det_face_info[0][1]), \
                    (self._det_face_info[0][2], self._det_face_info[0][3])
                # check if face is familiar, if not, colored as red, else colored as green
                if self._if_recognized:
                    color = [0, 255, 0]
                else:
                    color = [0, 0, 255]
                cv2.rectangle(frame, pos1, pos2, color, 2)
                # put self._recognized_name at the right corner of rectangle
                cv2.putText(frame, self._recognized_name, (pos1[0], pos1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow('frame', frame)

            # exit port
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # release
        self._release_resources()
        pass

    def start(self):
        process_thread = threading.Thread(target=self._image_processing)
        process_thread.start()
        self.video_capture()
        # stop background thread
        self._exit_thread = True


# ------------------------------------------------------------
def main():
    fr = Face_Recognition()
    fr.start()
    # fr.video_capture()
    pass


main()
