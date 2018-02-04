#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__date__ = '2018/01/22'


import cv2
import numpy as np
import tools
import serial


'''
This is a 'hello world' test for pan-tilt test.
Print circle_test.pdf and put the paper close to camera.
'''


def sent(transmission, x, y):
    if x > 320:
        yaw_angle = int(3050 - ((x - 320) / 320 * 3050))
    else:
        yaw_angle = int(3050 + ((320 - x) / 320 * 2650))
    if y < 240:
        pitch_angle = int(5000 - ((240 - y) / 240 * 5000))
    else:
        pitch_angle = int(5000 + ((y - 240) / 240 * 1000))
    # transmission.write(bytes.fromhex(yaw_angle))
    # transmission.write(bytes.fromhex(pitch_angle))
    mes = bytearray([0] * 9)
    mes[0] = 0xFA  # 250
    mes[1] = 0x04  # 004
    mes[2] = 0x00  # 000
    mes[3] = yaw_angle & 0xFF  # Take the last 8 bits
    mes[4] = (yaw_angle >> 8) & 0xFF  # Take the last 9 - 16 bits
    mes[5] = pitch_angle & 0xFF  # Take the last 8 bits
    mes[6] = (pitch_angle >> 8) & 0xFF  # Take the last 9 - 16 bits
    mes[7] = 0x00  # 000
    mes[8] = 0xFB  # 251
    print([yaw_angle, pitch_angle])  # Show yaw_angle and pitch_angle
    transmission.write(mes)


def func_detect_circle(frame, transmission):
    result = [0, 0, 0]
    # frame = cv2.resize(frame, (480, 360))
    b, g, r = cv2.split(frame)
    # sp = frame.shape
    # 480 640
    r = cv2.pyrDown(r)
    # Using Gaussian Pyramid to down-sample the red part of the frame we get,
    # thus, the following code could find the circles more efficient and accurate
    '''
    If you want see more about the Image Pyramids, then you might the following website
    https://docs.opencv.org/3.0-beta/doc/tutorials/imgproc/pyramids/pyramids.html?highlight=pyramids
    https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_pyramids/py_pyramids.html?highlight=pyramids
    '''
    _circles = cv2.HoughCircles(r, cv.CV_HOUGH_GRADIENT, 1.2, 1200,
                                param1=250, param2=80, minRadius=30, maxRadius=0)
    '''
    https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghcircles#houghcircles
    https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html?highlight=houghcircles
    '''
    if np.any(_circles):
        circles = _circles[0, :, :]
        circles = np.uint16(np.around(circles))
        for i in circles[:]:
            cv2.circle(frame, (2 * i[0], 2 * i[1]), 2 * i[2], (0, 0, 255), 5)
            # Draw the circles we get in the original image
            x = 0.0 + 2 * i[0]  # Get into the float form
            y = 0.0 + 2 * i[1]  # Get into the float form
            # Since we pyramid down the image,
            # we have to recalculate the center of the circle to fit in the original image,
            # that's why you can see 2 * i[0] sort of stuff
            sent(transmission, x, y)
            if i[2] > result[2]:
                result = list(i)
    cv2.imshow('Frame', frame)
    cv2.waitKey(1)
    # return result if sum(result) else []
    return [2 * i[0], 2 * i[1]] if sum(result) else []  # FIXME Could be wrong


t = serial.Serial('/dev/ttyUSB0', 115200)
tools.set_quit()
cap = cv2.VideoCapture(0)
while 1:
    ret, frame = cap.read()
    func_detect_circle(frame, t)
    #print (func_detect_circle(frame, t))