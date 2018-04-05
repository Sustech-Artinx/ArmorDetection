#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__date__ = '2018/04/06'

import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

from tools import quitable, folder
from detector import BarColor, Detector
from serial_msg import SerialWriter

# TODO: (x, y) -> (yaw, pitch)
#
# def get_converter(weight: int, height: int):
#     def f(n: float): # an odd function
#         return math.log(n + 1) if n >= 0 else math.log(1 - n)
#     def converter(x: int, y: int) -> tuple: # TODO
#         dx = x - weight // 2
#         dy = y - height // 2
#         return int(f(dx) * 8), int(f(dy) * 6)
#     return converter
#
# class Tracer:
#     """
#     (x, y) -> (delta_yaw, delta_pitch)
#     and keep movement smooth
#     """
#     def __init__(self, width, height):
#         self.step = 5
#         self.central_x = width >> 1
#         self.central_y = height >> 1
#
#     def convert(self, target: tuple) -> tuple:
#         if target is not None:
#             self.step = (self.step - 1) if self.step > 5 else 5
#             x, y = target
#             dx = x - self.central_x
#             dy = y - self.central_y
#             eta = self.step / math.sqrt(dx * dx + dy * dy)
#             return (-int(dx * eta), int(dy * eta)) if eta < 1 else target
#         else:
#             self.step = self.step + 1 if self.step < 50 else 50
#             return 0, 0


class Smoother:
    def __init__(self, shape):
        width, height = shape
        self.last_target = None
        self.invalid_count = 0
        self.far_enough = math.sqrt(width*width + height*height) / 12
        self.count_threshold = 5

    def smoothed(self, new_target):
        if new_target is None:
            if self.invalid_count > self.count_threshold:
                self.last_target = None
            else:
                self.invalid_count += 1
        else:
            if self.last_target is None:
                self.last_target = new_target
                self.invalid_count = 0
            else:
                nx, ny = new_target
                lx, ly = self.last_target
                dy, dx = ny - ly, nx - lx
                dis = math.sqrt(dx*dx + dy*dy)
                if dis > self.far_enough and self.invalid_count <= self.count_threshold:
                    self.invalid_count += 1
                else: # normal, or switch to new target
                    self.last_target = lx + dx//2, ly + dy//2
                    self.invalid_count = 0
        return self.last_target


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
    def run(self): # FIXME: move folder path to constructor
        file_list = folder("/home/jeeken/Pictures/DMovie", "jpg")
        for i in file_list:
            print("img_file:", i)
            mat = cv2.resize(cv2.imread(i), self.frame_size)
            target = self.detector.target(mat)
            print("target:", target)
            if plt.waitforbuttonpress():
                plt.close()
                break
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #    break


class CamCarTargetApp(CarTargetApp):
    def __init__(self, color: BarColor, msg: SerialWriter, cam_idx: int = 0,
                 frame_size: tuple = (480, 360), debug: bool = False):
        CarTargetApp.__init__(self, color, debug)
        self.debug = debug
        self.cap = cv2.VideoCapture(cam_idx)
        self.msg = msg
        self.frame_size = frame_size
        self.smoother = Smoother(frame_size)

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
        # TODO: unfinished
        while self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                break
            frame = CamCarTargetApp.undistort(cv2.resize(frame, self.frame_size))
            if self.debug and (cv2.waitKey(1) & 0xFF == ord('q')):
                break
            target = self.detector.target(frame)
            target_smoothed = self.smoother.smoothed(target)
            # TODO:
            # - convert coordinate
            # - communication
            # - debug helper


class VideoCarTargetApp(CarTargetApp):
    def __init__(self, color: BarColor, file: str, frame_size: tuple = (640, 480), debug: bool = False):
        CarTargetApp.__init__(self, color, debug)
        self.file = file
        self.debug = debug
        self.frame_size = frame_size
        self.smoother = Smoother(frame_size)

    @quitable
    def run(self):
        # out = cv2.VideoWriter(filename="/home/jeeken/Videos/180_out.mp4", fourcc=cv2.VideoWriter_fourcc(*"XVID"), fps=30.0, frameSize=(640, 480))
        cap = cv2.VideoCapture(self.file)
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, self.frame_size)
            target = self.detector.target(frame)
            target_smoothed = self.smoother.smoothed(target)
            print("target:  ", target)
            print("smoothed:", target_smoothed)
            print("--------------------")

            cv2.imshow("Original", frame)
            if target is not None:
                cv2.circle(img=frame, center=target, radius=4, color=(0, 255, 0), thickness=-1) # green, -1: filled
                aim_color = (0, 0, 255)
                x, y = target_smoothed
                cv2.circle(img=frame, center=target_smoothed, radius=20, color=aim_color, thickness=2) # red circle
                cv2.line(img=frame, pt1=(x-40, y), pt2=(x+40, y), color=aim_color, thickness=1)
                cv2.line(img=frame, pt1=(x, y-40), pt2=(x, y+40), color=aim_color, thickness=1)
            cv2.imshow("Aimed", frame)
            # out.write(frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()


if __name__ == "__main__":
    # TODO: get argument from command line
    # ser = SerialWriter(port='/dev/ttyUSB0', baudrate=115200)

    # app = CamCarTargetApp(color=BarColor.BLUE, msg=ser, cam_idx=1, frame_size=(480, 360), debug=True)

    # app = ImgCarTargetApp(color=BarColor.BLUE, frame_size=(480, 360), debug=True)
    # app = ImgCarTargetApp(color=BarColor.RED, frame_size=(480, 360), debug=True)

    # edit commented code if you want to debug Video mode
    app = VideoCarTargetApp(color=BarColor.BLUE, file="/home/jeeken/Videos/live_blue.avi", frame_size=(640, 480), debug=False)
    app.run()
