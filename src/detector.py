#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from enum import Enum


class BarColor(Enum):
    """Color of target bar"""
    BLUE = 0
    RED = 1


class TargetDis(Enum):
    """Distance of target: near, medium, far, ultra"""
    NEAR = 0
    MID = 1
    FAR = 2
    ULTRA = 3


# TODO: Detector for red light bars

class Detector:
    def __init__(self, color: BarColor, shape: tuple = (480, 360), debug: bool = False):
        self.color = color
        self.debug = debug
        self.distance = TargetDis.NEAR # for target_old

    def hsv(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # TODO: decide the range of lightness based on histogram
        if self.color == BarColor.BLUE:
            lower_light = np.array([90, 0, 215])
            upper_light = np.array([120, 100, 255])
            lower_halo  = np.array([90, 100, 185])
            upper_halo  = np.array([120, 255, 255])
        else: # self.color == BarColor.RED:
            lower_light = np.array([0, 0, 215])
            upper_light = np.array([25, 100, 255])
            # TODO
            lower_halo1 = np.array([170, 100, 185])
            upper_halo1 = np.array([180, 255, 255])
            lower_halo2 = np.array([0, 100, 185])
            upper_halo2 = np.array([25, 255, 255])
            lower_halo = cv2.bitwise_or(lower_halo1, lower_halo2)
            upper_halo = cv2.bitwise_or(upper_halo1, upper_halo2)

        light_area = cv2.inRange(hsv_frame, lower_light, upper_light)
        halo_area = cv2.inRange(hsv_frame, lower_halo, upper_halo)
        return light_area, halo_area

    @staticmethod
    def get_connected_components_info(frame):
        cpn_count, label_map, cpn_info, centroids = \
            cv2.connectedComponentsWithStats(image=frame, connectivity=4, ltype=cv2.CV_32S)

        def pack(label, info, centroid):
            northwest_x, northwest_y, width, height, pixel_number = info
            x, y = centroid
            return {
                "label": label,
                "size": (width, height),
                "centroid": (int(round(x)), int(round(y))),
                "northwest": (northwest_x, northwest_y),
                "southeast": (northwest_x + width, northwest_y + height),
                "pixels": pixel_number
            }
        # background removed
        return sorted(map(pack, range(0, cpn_count), cpn_info, centroids), key=lambda x: -x["pixels"])[1:], label_map

    @staticmethod
    def zoom_and_shift(area):
        nw_x, nw_y = area["northwest"]
        se_x, se_y = area["southeast"]
        nw_x -= area["width"] // 4
        se_x += area["width"] // 4
        nw_y -= area["height"] // 2
        return {
            "northwest": (nw_x, nw_y), "southeast": (se_x, se_y),
            "width": se_x - nw_x, "height": se_y - se_x
        }

    @staticmethod
    def select_lights_in_halo(light, halo):
        light_components, light_map = Detector.get_connected_components_info(light)
        halo_components= Detector.get_connected_components_info(halo)[0]
        selected_halo = list(filter(lambda c: c["pixels"] > 20, halo_components)) # TODO: eliminate magic number

        # find halo area
        try:
            northwest_x = min([x["northwest"][0] for x in selected_halo])
            northwest_y = min([x["northwest"][1] for x in selected_halo])
            southeast_x = max([x["southeast"][0] for x in selected_halo])
            southeast_y = max([x["southeast"][1] for x in selected_halo])
        except ValueError: # selected_halo is []
            return [], None
        width = southeast_x - northwest_x
        height = southeast_y - northwest_y

        # zoom and shift
        northwest_x -= width // 4
        southeast_x += width // 4
        northwest_y -= height // 2

        # select inside ones
        def is_fully_inside(c):
            c_nw_x, c_nw_y = c["northwest"]
            c_se_x, c_se_y = c["southeast"]
            return northwest_x <= c_nw_x and northwest_y <= c_nw_y and c_se_x <= southeast_x and c_se_y <= southeast_y
        insides = list(filter(is_fully_inside, light_components)) # FIXME: why doesn't work if don't convert it to a list?

        # add angle info
        for i in insides:
            nw_x, nw_y = i["northwest"]
            se_x, se_y = i["southeast"]
            tx, ty = nw_x + np.argmax(light_map[nw_y, nw_x:se_x]), nw_y     # top peak
            bx, by = nw_x + np.argmax(light_map[se_y-1, nw_x:se_x]), se_y-1     # bottom peak
            angle = math.atan2(by-ty, tx-bx) / math.pi  # unit: pi rad
            i["angle"] = angle

        return insides, {"northwest": (northwest_x, northwest_y), "southeast": (southeast_x, southeast_y)}

    @staticmethod
    def select_valid_bars(lights):
        if lights == []:
            return []

        # eliminate small ones
        max_pixels = max([x["pixels"] for x in lights])
        big_enough = list(filter(lambda x: (x["pixels"] / max_pixels) > 0.06, lights))

        def is_bar_near(bar):
            width, height = bar["size"]
            square_ratio = height / width
            return 1.5 <= square_ratio < 8
        # def is_bar_far(bar):
        #     width, height = bar["size"]
        #     square_ratio = height / width
        #     return 1.3 < square_ratio < 4
        selected = list(filter(is_bar_near, big_enough))
        # if len(selected) < 2:
        #     selected = list(filter(is_bar_far, big_enough))

        vertical_bars = filter(lambda x: abs(x["angle"] - 0.5) < 0.2, selected)

        # filled_ratio test for removing components in strange shape
        def filled_ratio(bar):
            width, height = bar["size"]
            return -bar["pixels"] / (width * height)
        return sorted(vertical_bars, key=filled_ratio)[0:6]

    @staticmethod
    def select_pair(bars):
        length = len(bars)
        if length < 2:
            return ()

        pairs = [(i, j) for i in range(0, length-1) for j in range(i+1, length)]

        def y_dis(bar1, bar2):
            y1 = bar1["centroid"][1]
            y2 = bar2["centroid"][1]
            height_base = min((bar1["size"][1], bar2["size"][1]))
            return abs(y1 - y2) / height_base

        def area(bar1, bar2):
            a1 = bar1["pixels"]
            a2 = bar2["pixels"]
            return abs(a1 - a2) / min((a1, a2))

        def parallel(bar1, bar2):
            theta1, theta2 = bar1["angle"], bar2["angle"]
            return abs(theta1 - theta2) * 5

        def judge(pair):
            index1, index2 = pair
            bar1, bar2 = bars[index1], bars[index2]
            policy = [(y_dis, 0.6), (area, 0.3), (parallel, 1.2)]
            # # debug
            # print(pair, 'y_dis:', y_dis(bar1, bar2), '\tarea:', area(bar1, bar2), '\tparallel:', parallel(bar1, bar2))
            return sum([func(bar1, bar2) * coefficient for func, coefficient in policy])

        judging_result = [{"indices": pair, "score": judge(pair)} for pair in pairs]
        winner = min(judging_result, key=lambda x: x["score"])

        # # debug
        # print("winner:", winner)
        if winner["score"] > 1.2:
            return ()

        i1, i2 = winner["indices"]
        return bars[i1], bars[i2]

    def target(self, frame):
        # frame = cv2.blur(frame, ksize=(4, 4))
        frame = cv2.pyrUp(cv2.pyrDown(frame)) # down-sample
        light, halo = self.hsv(frame)
        lights_in_halo, selected_area = Detector.select_lights_in_halo(light, halo)
        selected_bars = Detector.select_valid_bars(lights_in_halo)
        selected_pair = Detector.select_pair(selected_bars)
        if selected_pair != ():
            bar1, bar2 = selected_pair
            def mid_point(pt1, pt2):
                x1, y1 = pt1
                x2, y2 = pt2
                return (x1 + x2) // 2, (y1 + y2) // 2
            target = mid_point(bar1["centroid"], bar2["centroid"])
        else:
            target = None

        # # debug
        # for each in selected_pair:
        #     cv2.circle(img=frame, center=each["centroid"], radius=7, color=(0,255,0), thickness=2)
        # cv2.imshow('Debug', frame)

        if self.debug:
            show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            plt.figure("Car Detection Debug")

            plt.subplot(2, 2, 1)
            plt.title("Original")
            plt.imshow(show)
            plt.axis('off')

            if selected_area is not None:
                cv2.rectangle(halo, pt1=selected_area["northwest"], pt2=selected_area["southeast"], color=255,
                              thickness=1)
                cv2.rectangle(light, pt1=selected_area["northwest"], pt2=selected_area["southeast"], color=255,
                              thickness=1)
            for each in selected_bars:
                cv2.circle(img=show, center=each["centroid"], radius=7, color=(0, 255, 0), thickness=2)
            for each in selected_pair:
                cv2.circle(img=show, center=each["centroid"], radius=5, color=(255, 0, 0), thickness=2)

            plt.subplot(2, 2, 2)
            plt.title("halo")
            plt.imshow(halo)
            plt.axis('off')

            plt.subplot(2, 2, 3)
            plt.title("light")
            plt.imshow(light)
            plt.axis('off')

            plt.subplot(2, 2, 4)
            plt.title("selected")
            plt.imshow(show)
            plt.axis('off')

        return target
