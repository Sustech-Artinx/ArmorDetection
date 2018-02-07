#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__date__ = '2018/01/22'


import cv2
import numpy as np
import time
import math

from tools import quitable, folder
from detector import BarColor, Detector
from serial_msg import SerialWriter


def get_converter(weight: int, height: int):
    def f(n: float): # an odd function
        return math.log(n + 1) if n >= 0 else math.log(1 - n)
    def converter(x: int, y: int) -> tuple: # TODO
        dx = x - weight // 2
        dy = y - height // 2
        return int(f(dx) * 8), int(f(dy) * 6)
    return converter


class CarTargetApp:
    def __init__(self, color: BarColor, debug: bool = False):
        self.detector = Detector(color=color, debug=debug)

    def __del__(self):
        cv2.destroyAllWindows()

    @quitable
    def run(self):
        pass


class ImgCarTargetApp(CarTargetApp):
    def __init__(self, color: BarColor, frame_size: tuple, debug: bool = False):
        CarTargetApp.__init__(self, color, debug)
        self.frame_size = frame_size

    @quitable
    def run(self): # FIXME
        file_list = folder("/home/jeeken/Projects/CV_tools/img/target", "png")
        for i in file_list:
            print(i)
            mat = cv2.resize(cv2.imread(i), self.frame_size)
            target = self.detector.target(mat) # TODO
            print(target)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break


class CamCarTargetApp(CarTargetApp):
    def __init__(self, color: BarColor, msg: SerialWriter, cam_idx: int = 0,
                 frame_size: tuple = (480, 360), debug: bool = False):
        CarTargetApp.__init__(self, color, debug)
        self.debug = debug
        self.cap = cv2.VideoCapture(cam_idx)
        self.msg = msg
        self.frame_size = frame_size
        self.converter = get_converter(*frame_size)

    def __del__(self):
        CarTargetApp.__del__(self)
        self.cap.release()

    @staticmethod
    def undistort(mat):
        """
        undistort a frame
        :param mat: a frame
        :return: undistort frame
        """
        h, w = mat.shape[:2]
        # camera undistort matrix
        mtx = np.array([[544.78014225, 0., 332.28614309],
                        [0., 541.53884466, 241.76573558],
                        [0., 0., 1.]])
        dist = np.array([[-4.35436872e-01, 2.13933541e-01, 4.09271605e-04, 5.63531212e-03, -6.74471459e-03]])
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
        return cv2.undistort(mat, mtx, dist, None, new_camera_mtx)

    @quitable
    def run(self):
        while True:
            if self.cap.isOpened():
                success, mat = self.cap.read()
                if success:
                    mat = CamCarTargetApp.undistort(cv2.resize(mat, self.frame_size))
                else:
                    break
            else:
                break

            if self.debug and (cv2.waitKey(1) & 0xFF == ord('q')):
                break

            time.sleep(0.015)
            target = self.detector.target(mat)
            if target is not None:
                x, y = target
                # y -= 104 # camera coordinate -> gun coordinate (Remark: camera !// gun )
                self.msg.write(*self.converter(x, y))


class VideoCarTargetApp(CarTargetApp):
    def __init__(self, color: BarColor, debug: bool = False):
        CarTargetApp.__init__(self, color, debug)

    @quitable
    def run(self): # FIXME
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter('test.avi', fourcc, 20.0, (640, 480))
        cap = cv2.VideoCapture('red_car.avi')
        skip = 0
        count = -1
        wait = 10 if skip == 0 else 0
        while cap.isOpened():
            count += 1
            ret, mat = cap.read()
            # FIXME: skip is not modified during the process, how will the process terminate?
            if skip == 0:
                print("frame " + str(count))
                target = self.detector.target(mat) # TODO
                k = cv2.waitKey(wait)
                if k & 0xFF == ord('q'):
                    # out.release()
                    break
                if k & 0xFF == ord('p'):
                    wait = 0
                if k & 0xFF == ord('c'):
                    wait = 10
            else:
                skip -= 1
        cap.release()


if __name__ == "__main__":
    # TODO: get argument from command line
    ser = SerialWriter(port='/dev/ttyUSB0', baudrate=115200)
    app = CamCarTargetApp(color=BarColor.BLUE, msg=ser, cam_idx=0, debug=True)
    # app = ImgCarTargetApp(color=BarColor.BLUE, frame_size=(480, 360), debug=True)
    app.run()
