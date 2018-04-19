#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import math
import numpy as np
from enum import Enum
from random import randint


# class BgrColor(Enum):
#     RED = (0, 0, 255)
#     BLUE = (255, 0, 0)
#     GREEN = (0, 255, 0)
#     YELLOW = (0, 255, 255)
#     PURPLE = (255, 0, 255)


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


def round_point(pt):
    x, y = pt
    return (round(x), round(y))


def mid_point(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return (x1 + x2) // 2, (y1 + y2) // 2


class Detector:
    # TODO: remove shape from self.__init__
    def __init__(self, color: BarColor, shape: tuple = (480, 360), debug = False):
        self.color = color
        self.refresh(frame=None, debug=debug)

    def refresh(self, frame: np.ndarray, debug=None):
        self.frame = frame
        if debug is not None:
            self.debug = debug
        self.debug_imgs = []

    def add_debug_img(self, title: str, img):
        if self.debug:
            if callable(img):
                self.debug_imgs.append((title, img()))
            else:
                self.debug_imgs.append((title, img))

    def debug_print(self, *msg):
        if self.debug:
            print("Debug>>>", *msg)
            # print("Debug>>>", *msg, file=sys.stderr)

    def hsv(self, frame: np.ndarray):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if self.color == BarColor.RED:
            h, s, v = cv2.split(hsv_frame.astype(np.uint16))
            h += 107
            h %= 180
            hsv_frame = cv2.merge((h, s, v)).astype(np.uint8)
            self.add_debug_img("Red to Blue", cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR))

        # TODO: decide the range of lightness based on histogram
        # lower_light = np.array([90, 0, 215])
        lower_light = np.array([90, 0, 210])
        upper_light = np.array([120, 100, 255])
        lower_halo  = np.array([90, 100, 185])
        upper_halo  = np.array([120, 255, 255])

        light_area = cv2.inRange(hsv_frame, lower_light, upper_light)
        halo_area = cv2.inRange(hsv_frame, lower_halo, upper_halo)
        return light_area, halo_area

    @staticmethod
    def get_connected_components_info(frame):
        # cpn: component
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

    # @staticmethod
    # def select_lights_in_halo(light, halo):
    #     light_components, light_map = Detector.get_connected_components_info(light)
    #     halo_components= Detector.get_connected_components_info(halo)[0]
    #     selected_halo = list(filter(lambda c: c["pixels"] > 20, halo_components)) # TODO: eliminate magic number
    #
    #     # find halo area
    #     try:
    #         northwest_x = min([x["northwest"][0] for x in selected_halo])
    #         northwest_y = min([x["northwest"][1] for x in selected_halo])
    #         southeast_x = max([x["southeast"][0] for x in selected_halo])
    #         southeast_y = max([x["southeast"][1] for x in selected_halo])
    #     except ValueError: # selected_halo is []
    #         return [], None
    #     width = southeast_x - northwest_x
    #     height = southeast_y - northwest_y
    #
    #     # zoom and shift
    #     northwest_x -= width // 4
    #     southeast_x += width // 4
    #     northwest_y -= height // 2
    #
    #     # select inside ones
    #     def is_fully_inside(c):
    #         c_nw_x, c_nw_y = c["northwest"]
    #         c_se_x, c_se_y = c["southeast"]
    #         return northwest_x <= c_nw_x and northwest_y <= c_nw_y and c_se_x <= southeast_x and c_se_y <= southeast_y
    #     insides = list(filter(is_fully_inside, light_components)) # FIXME: why doesn't work if don't convert it to a list?
    #
    #     # add angle info
    #     for i in insides:
    #         nw_x, nw_y = i["northwest"]
    #         se_x, se_y = i["southeast"]
    #         top_line, bottom_line = light_map[nw_y, nw_x:se_x], light_map[se_y-1, nw_x:se_x]
    #         tx, ty = nw_x + np.argmax(top_line), nw_y           # top peak
    #         bx, by = nw_x + np.argmax(bottom_line), se_y-1      # bottom peak
    #         angle = math.atan2(by-ty, tx-bx) / math.pi          # unit: pi rad
    #         i["angle"] = angle
    #
    #     return insides, {"northwest": (northwest_x, northwest_y), "southeast": (southeast_x, southeast_y)}

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
        def is_bar_far(bar):
            width, height = bar["size"]
            square_ratio = height / width
            return 0.8 < square_ratio < 2
        selected = list(filter(is_bar_near, big_enough))
        if len(selected) < 2:
            selected = list(filter(is_bar_far, big_enough))

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

        def square_ratio(bar1, bar2): # TODO
            pass

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

    def get_lights(self, binary_img: np.ndarray):
        _, width = binary_img.shape
        # https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours
        img, contours, hierarchy = cv2.findContours(binary_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        rectangles_info = [cv2.minAreaRect(i) for i in contours if len(i) * 60 > width]  # filter (omit small ones) and map
        # rectangles_info = [cv2.fitEllipse(i) for i in contours if len(i) * 60 > width]
        return rectangles_info

    def halo_circle(self, binary_img: np.ndarray):
        height, width = binary_img.shape
        img, contours, hierarchy = cv2.findContours(binary_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        selected_contours = [i for i in contours if len(i) * 60 > width]  # small contours omitted
        if len(selected_contours) > 0:
            combined_halo_points = np.concatenate(selected_contours)
            original_center, original_radius = cv2.minEnclosingCircle(combined_halo_points)
            x, y = original_center
            center = round(x), round(y - original_radius * 0.6)  # shift up
            radius = round(original_radius * 1.8 if original_radius < (width / 4) else width * 0.375)  # zoom
        else: # no valid halo, consider the central area
            center = width // 2, height // 2
            radius = width // 3
        return center, radius

    def select_lights_in_halo(self, lights, halo_center: tuple, halo_radius):
        x, y = halo_center
        def is_inside(light):
            (xl, yl), shape, angle = light
            return (xl-x)**2 + (yl-y)**2 < halo_radius**2
        return filter(is_inside, lights)

    def select_vertical_lights(self, lights):
        def normalize_angle(rec):
            center, shape, angle = rec
            w, h = shape
            if w > h:
                return center, (h, w), angle+90
            else:
                return rec

        def is_vertical(rec):
            center, shape, angle = rec
            return -20 < angle < 20

        angle_normalized_lights = map(normalize_angle, lights)
        vertical_lights = [i for i in angle_normalized_lights if is_vertical(i)]

        # debug
        def get_img_vertical_lights():
            show_lights = np.copy(self.frame)
            for i in vertical_lights:
                center, shape, angle = i
                rand_color = (randint(40, 230), randint(40, 255), randint(40, 255))
                cv2.ellipse(show_lights, center=round_point(center), axes=round_point(shape), angle=angle,
                            startAngle=0, endAngle=360, color=rand_color, thickness=2)
            return show_lights
        self.add_debug_img("Vertical Lights", get_img_vertical_lights)
        return vertical_lights

    def get_armor(self, lights):
        lights = list(lights)
        lights_quantity = len(lights)

        if lights_quantity < 2:
            return None

        pairs = [(i, j) for i in range(0, lights_quantity - 1) for j in range(i + 1, lights_quantity)]

        def parallel(light1, light2):
            angle1, angle2 = light1[2], light2[2]
            return abs(angle1 - angle2) / 20

        def size_similarity(light1, light2):
            (w1, h1), (w2, h2) = light1[1], light2[1]
            area1, area2 = w1 * h1, w2 * h2
            min_size = area1 if area1 < area2 else area2
            return abs(area1 - area2) / min_size

        def square_ratio(light1, light2):
            (x1, y1), (w1, h1), angle1 = light1
            (x2, y2), (w2, h2), angle2 = light2
            ratio = 2 * math.sqrt((x1-x2)**2 + (y1-y2)**2) / (h1+h2)
            return abs(ratio - 2.4) ** 2 if ratio > 0.85 else 1e9

        def y_dis(light1, light2):
            (x1, y1), shape1, angle1 = light1
            (x2, y2), shape2, angle2 = light2
            min_y = y1 if y1 < y2 else y2
            return abs(y1 - y2) / min_y

        def judge(pair):
            i1, i2 = pair
            bar1, bar2 = lights[i1], lights[i2]
            policy = [(square_ratio, 5), (y_dis, 15), (size_similarity, 1), (parallel, 1)]  # [(func, coefficient),...]
            return sum((func(bar1, bar2) * coefficient for func, coefficient in policy))  # weighted sum

        judge_results = [(pair, judge(pair)) for pair in pairs]
        (i1, i2), winner_score = min(judge_results, key=lambda x: x[1])

        if winner_score > 5:
            return None

        def get_img_selected_pair():
            selected_pair_show = np.copy(self.frame)
            l1, l2 = lights[i1], lights[i2]
            self.debug_print("l1:", l1)
            self.debug_print("l2:", l2)
            self.debug_print("score:", winner_score)
            self.debug_print("parallel:", parallel(l1, l2))
            self.debug_print("size_similarity:", size_similarity(l1, l2))
            self.debug_print("square_ratio:", 5 * square_ratio(l1, l2))
            self.debug_print("y_dis:", 15 * y_dis(l1, l2))
            c1, s1, a1 = l1
            c2, s2, a2 = l2
            cv2.drawMarker(selected_pair_show, position=round_point(c1), color=(0, 255, 255), markerSize=15, thickness=2)
            cv2.drawMarker(selected_pair_show, position=round_point(c2), color=(0, 255, 255), markerSize=15, thickness=2)
            return selected_pair_show
        self.add_debug_img("Selected Pair", get_img_selected_pair)

        x1, y1 = lights[i1][0]
        x2, y2 = lights[i2][0]
        return round((x1+x2) * 0.5), round((y1+y2) * 0.5)

    def target(self, frame: np.ndarray):
        self.refresh(frame)

        # frame = cv2.blur(frame, ksize=(4, 4))
        frame = cv2.pyrUp(cv2.pyrDown(frame)) # down-sample
        light, halo = self.hsv(frame)

        # lights_in_halo, selected_area = Detector.select_lights_in_halo(light, halo)

        halo_center, halo_radius = self.halo_circle(halo)
        lights_info = self.get_lights(light)

        # only draw on light/halo after info got, or draw on a copied mat
        def get_img_halo():
            cv2.circle(img=halo, center=halo_center, radius=halo_radius, color=255, thickness=2)
            return halo
        def get_img_light():
            cv2.circle(img=light, center=halo_center, radius=halo_radius, color=255, thickness=2)
            return light
        self.add_debug_img("Halo", get_img_halo)
        self.add_debug_img("Light", get_img_light)

        lights_inside = self.select_lights_in_halo(lights_info, halo_center, halo_radius)
        vertical_lights = self.select_vertical_lights(lights_inside)
        target = self.get_armor(vertical_lights)

        # selected_bars = Detector.select_valid_bars(lights_in_halo)
        # selected_pair = Detector.select_pair(selected_bars)
        # if selected_pair != ():
        #     bar1, bar2 = selected_pair
        #     target = mid_point(bar1["centroid"], bar2["centroid"])
        # else:
        #     target = None
        #
        # # def get_img_selected_halo():
        # #     cv2.rectangle(halo, pt1=selected_area["northwest"], pt2=selected_area["southeast"], color=255, thickness=1)
        # #     return halo
        # # def get_img_selected_light():
        # #     cv2.rectangle(light, pt1=selected_area["northwest"], pt2=selected_area["southeast"], color=255, thickness=1)
        # #     return light
        # # if selected_area is not None:
        # #     self.add_debug_img("Halo", get_img_selected_halo)
        # #     self.add_debug_img("Light", get_img_selected_light)

        # def get_img_selected():
        #     selected = np.copy(self.frame)
        #     for each in selected_bars:
        #         cv2.drawMarker(img=selected, position=each["centroid"], color=(0, 255, 255), markerSize=25, thickness=2)
        #     for each in selected_pair:
        #         cv2.circle(img=selected, center=each["centroid"], radius=5, color=(0, 0, 255), thickness=-1)
        #     return selected
        # self.add_debug_img("Selected", get_img_selected)

        return target
