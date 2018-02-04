#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__date__ = '2018/01/22'


import signal
import sys
import cv2


def quit(signum, frame):
    print("TOOL : interupt " + str(signum) + " , program terminate")
    sys.exit()


def set_quit():
    '''
    Set Ctrl-C signal to quit.
    '''
    signal.signal(signal.SIGINT, quit)


# A decorator
def quitable(func):
    def f(*args, **kwargs):
        signal.signal(signal.SIGINT, quit)
        func(*args, **kwargs)
    return f


def folder(file_dir, file_type):
    '''
    Return a list of a type of files in a folder.
    
    Args:
        file_dir: A folder to find file list.
        file_type: A type of file.
    
    Rerturns:
        A list of string with absolute path.
    '''
    import os
    file_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == "." + file_type:
                file_list.append(os.path.join(root, file))
    return file_list


def mouth_callback_show_pix(event, x, y, a, b):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("x, y, b, g, r " + str(x) + " " + str(y) + " " + str(b[y, x]))


def set_mouth_callback_show_pix(window, mat):
    '''
    Set mouth callback for a window and a mat. Left click a pixel and consol will print [x, y, b, g, r] 
    
    Args:
        window: A window to set mat callback
        mat: A mat to print pixel
    '''
    cv2.setMouseCallback(window, mouth_callback_show_pix, mat)
