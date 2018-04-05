#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import copy
import math
import numpy as np
from matplotlib import pyplot as plt
from enum import Enum
from tools import set_mouth_callback_show_pix


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
        # self.min_step = 5 # FIXME Unknown Use # Never used before reassigned a new value
        # self.target_last = np.array([240, 180], dtype=np.int32) # suggest setting the central point as the initial value

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


    def track(self, mat, color_threshold, square_ratio, angle_rate, length_rate, matrix_threshold):
        """
        1. color threshold
        2. segment (connected component), get bar
           delete ultra small and large
        3. match bar, get bar pairs
           similar angle and length
        4. select bar pairs
              |       /
             |     / |
            |    /  |
           | z /   |
          | /     |  y
         |/    θ|
        --------
            x
        60 < theta < 120
        1.3 < x / y < 3
        :param mat: a frame
        :param color_threshold: value of color threshold
        :param square_ratio: delete connect component which height / width < square_ratio, delete too "fat"
        :param angle_rate: bar match angle rate factor
        :param length_rate: bar match length rate factor
        :param matrix_threshold: match pairs by rate score > matrix_threshold
        :return: None when no target, (x, y, pixel_count) for target
        """
        # FIXME

        # Relative threshold
        # set 255 if b - r < color_threshold
        b, g, r = cv2.split(mat)
        b = np.asarray(b, dtype='int32')
        r = np.asarray(r, dtype='int32')
        if self.color == BarColor.BLUE:
            b = (b - r) < color_threshold
        elif self.color == BarColor.RED:
            # more strict threshold condition for red car
            b = (~(((r - b) > color_threshold) & (g < 100) & (r > 100)))
        b = b.astype(np.uint8) * 255

        if self.debug:
            cv2.imshow('threshold', b)

        # Label Connected Component
        connect_output = cv2.connectedComponentsWithStats(b, 4, cv2.CV_32S)
        # connect_output: tuple (int component_count, ndarray label_map, ndarray connect_component_info, ndarray unused_not_clear)
        # label_map: use int to label connect component, same int value means one component, size equal to "b"
        # connect_component_info: a n*5 ndarray showing connect component info, [left_top_x, left_top_y, width, height, pixel_number]

        # Delete Component according to Height / Width >= 3
        connect_label = connect_output[1]
        # connect_data = [[leftmost (x), topmost (y), horizontal size, vertical size, total area in pixels], ...]
        connect_data = connect_output[2]
        connect_data[connect_data[:, 0] >= mat.shape[1]] = 0  # clear connected components out of bound
        connect_data[connect_data[:, 1] >= mat.shape[0]] = 0  # clear connected components out of bound

        if self.debug:
            print("connected components num: " + str(len(connect_output[2])))

            # print("square_scale: " + str(connect_data[:, 3] / connect_data[:, 2]))
            connect_max_index = np.argmax(connect_data[:, 4])
            connect_label_show = copy.deepcopy(connect_label).astype(np.uint8)
            if connect_max_index != 0:
                connect_label_show[connect_label == connect_max_index] = 0
                connect_label_show[connect_label == 0] = connect_max_index
            connect_label_show = cv2.equalizeHist(connect_label_show)
            for i in range(len(connect_data)):
                cv2.rectangle(connect_label_show, (connect_data[i][0] - 1, connect_data[i][1] - 1),
                              (connect_data[i][0] + connect_data[i][2] + 1, connect_data[i][1] + connect_data[i][3] + 1),
                              155, 1)
                cv2.putText(connect_label_show, str(i), (connect_data[i][0] - 5, connect_data[i][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            0, 1, cv2.LINE_AA)
            cv2.imshow('Component', connect_label_show)

        def get_delete_strict_list(connect_data, square_ratio):
            """
            strict delete connect component, used when object is near
            :param connect_data:
            :param square_ratio:
            :return: index of connect_data need to delete
            """
            # delete vertical / horizontal < square_scale partitions
            connect_ratio_delete_list = np.where((connect_data[:, 3] / connect_data[:, 2]) < square_ratio)[0]
            # delete area < 30 or > 200
            connect_size_delete_list = np.where((connect_data[:, 4] < 30) | (connect_data[:, 4] > 200))[0]
            connect_delete_list = np.hstack((connect_ratio_delete_list, connect_size_delete_list))

            if self.debug:
                connect_label_delete_show = copy.deepcopy(connect_label).astype(np.uint8)
                for i in connect_delete_list:
                    connect_label_delete_show[connect_label_delete_show == i] = 0
                connect_label_delete_show = cv2.equalizeHist(connect_label_delete_show)
                cv2.imshow('Delete Component', connect_label_delete_show)
            return connect_delete_list

        def get_delete_loose_list(connect_data, square_ratio):
            """
            get_delete_loose_list seems the same as get_delete_strict_list.
            Actually I want to distinguish near and far by calling two diff. function,
            but later I decided to call track() with diff. square_ratio
            :param connect_data:
            :param square_ratio:
            :return:
            """
            connect_ratio_delete_list = np.where((connect_data[:, 3] / connect_data[:, 2]) < square_ratio)[0]
            # delete area < 5
            connect_size_delete_list = np.where((connect_data[:, 4] < 30) | (connect_data[:, 4] > 3000))[0] # why 3000 ?
            connect_delete_list = np.hstack((connect_ratio_delete_list, connect_size_delete_list))

            if self.debug:
                connect_label_delete_show = copy.deepcopy(connect_label).astype(np.uint8)
                for i in connect_delete_list:
                    connect_label_delete_show[connect_label_delete_show == i] = 0
                connect_label_delete_show = cv2.equalizeHist(connect_label_delete_show)
                cv2.imshow('Delete Component', connect_label_delete_show)
            return connect_delete_list

        if self.distance == TargetDis.NEAR:
            connect_delete_list = get_delete_strict_list(connect_data, square_ratio)
        elif self.distance == TargetDis.MID:
            connect_delete_list = get_delete_loose_list(connect_data, square_ratio)
        else:
            connect_delete_list = []

        connect_remain = []
        # for i in range(len(connect_data)):
        #     if i not in connect_delete_list:
        #         connect_remain.append(connect_data[i])
        # TODO: consider whether we can improve the algorithm ( O(n^2) -> O(nlogn) )
        for each in connect_data:
            if each not in connect_delete_list:
                connect_remain.append(each)

        # no target return None if all partitions are deleted
        if len(connect_remain) < 2:
            return None

        # Get peak_point points in each light bar
        # FIXME:
        # This is not a good implement.
        # We can first crop this component from label map according to component info
        # This use something like np.argmin, np.argmax
        bar_peak_point = []
        for each in connect_remain: # connect_remain: data tuple of remained connecetd components
            top_y = each[1] # top y coordinate of connecetd components
            top_x_series = np.where(connect_label[top_y+1, each[0]:each[0]+each[2]] != 0)[0]
            # locate a sereis of x coordinate of the light bar
            # Unsure: why first y, then x ?
            # Why only keep index[0]? Because np.where(condition) returns a tuple which has only one element
            if len(top_x_series) == 0:
                return None
            n1 = int((np.max(top_x_series) + np.min(top_x_series)) / 2 + each[0]) # x coordinate of mid point of top line of light bar
            down_y = each[1] + each[3] - 1 # y coordinate of bottom line of light bar
            down_x_series = np.where(connect_label[down_y-1, each[0]: each[0]+each[2]] != 0)[0]
            # series of x coordinate of bottom line of light bar
            if len(down_x_series) == 0:
                return None
            n2 = int((np.max(down_x_series) + np.min(down_x_series)) / 2 + each[0])
            # x coordinate of mid point of bottom line of light bar s
            bar_peak_point.append([n1, top_y, n2, down_y, each[4]])
            # Should be [top_mid_x, top_mid_y, down_mid_x, down_mid_y, pixel count]
        # [[top_left_x, top_left_y, down_right_x, down_right_y, pixel count], ...]
        bar_peak_point = np.array(bar_peak_point)

        if self.debug:
            for each in bar_peak_point:
                cv2.circle(mat, (each[0], each[1]), 3, (0, 0, 255), -1)
                cv2.circle(mat, (each[2], each[3]), 3, (0, 0, 255), -1)
            for i in range(len(connect_remain)):
                cv2.putText(mat, str(i), (connect_remain[i][0] - 5, connect_remain[i][1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Calculate bar bar_length, list
        bar_length = np.sqrt(np.power((bar_peak_point[:, 3] - bar_peak_point[:, 1]), 2) +
                             np.power((bar_peak_point[:, 2] - bar_peak_point[:, 0]), 2))
        if self.debug:
            print("bar_length" + str(bar_length))

        # Calculate bar_length matrix
        # not good implement, should use numpy
        # although we only need upper triangular matrix,
        # but it would be faster is we ignore redundant lower part and use numpy
        matrix_bar_length_diff = []
        for i in range(len(bar_length)):
            temp_one_line = []
            for j in range(len(bar_length)):
                if j <= i:
                    temp_one_line.append(0)  # only upper triangular part of the matrix is calculated
                elif abs(bar_length[i] - bar_length[j]) == 0:
                    temp_one_line.append(1)
                else:
                    temp_one_line.append(abs(bar_length[i] - bar_length[j]))
            matrix_bar_length_diff.append(temp_one_line)
        matrix_bar_length_diff = np.array(matrix_bar_length_diff)  # get an array of absoulte bar length difference
        if self.debug:
            print("matrix_bar_length_diff")
            for i in range(len(matrix_bar_length_diff)):
                print(matrix_bar_length_diff[i])

        # Calculate bar angle
        bar_angle = []
        for i in range(len(bar_peak_point)):
            if bar_peak_point[i][2] - bar_peak_point[i][0] == 0:
                bar_angle.append(90)
            else:
                arc = math.atan(float(bar_peak_point[i][3] - bar_peak_point[i][1]) /
                                float(bar_peak_point[i][2] - bar_peak_point[i][0])) * 180 / math.pi
                arc = arc if arc > 0 else arc + 180  # phase shift to make it the same angle for right bar and left bar
                bar_angle.append(arc)
        if self.debug:
            print("bar_angle" + str(bar_angle))

        # Calculate angle matrix
        matrix_bar_angle_diff = []
        for i in range(len(bar_angle)):  # bar_angle : all bar angles
            temp_one_line = []
            for j in range(len(bar_angle)):
                if j <= i:
                    temp_one_line.append(0)
                else:
                    temp_one_line.append(abs(bar_angle[i] - bar_angle[j]))
                    # get bar angle difference for j > i
                    # FIXME why necessary ?
            matrix_bar_angle_diff.append(temp_one_line)  # a list of abs difference of bar angles
        matrix_bar_angle_diff = np.array(matrix_bar_angle_diff)  # make the list an array
        if self.debug:
            print("matrix_bar_angle_diff")
            for i in range(len(matrix_bar_angle_diff)):
                print(matrix_bar_angle_diff[i])

        # Calculate weighted sum matrix
        matrix_bar_sum_diff = angle_rate * matrix_bar_angle_diff + length_rate * matrix_bar_length_diff
        if self.debug:
            print("matrix_bar_sum_diff")
            for i in range(len(matrix_bar_sum_diff)):
                print(matrix_bar_sum_diff[i])

        # select pairs by threshold
        matrix_bar_sum_diff_threshold = np.zeros(matrix_bar_sum_diff.shape)
        # set corresponding position to 1
        matrix_bar_sum_diff_threshold[(matrix_bar_sum_diff < matrix_threshold) & (matrix_bar_sum_diff > 0)] = 1
        if len(np.where(matrix_bar_sum_diff_threshold == 1)[0]) == 0:
            return None

        """
               /|
           z /  |
           /    |  y
         /    θ |
        --------
            x
           60 < theta < 120
        """
        # bar pair x distance
        matrix_bar_distance_x = np.zeros(matrix_bar_sum_diff_threshold.shape)
        for i in range(len(matrix_bar_distance_x)):
            for j in range(len(matrix_bar_distance_x)):
                if i < j:  # bar_peak_point [[top_left_x, top_left_y, down_right_x, down_right_y, pixel count], ...]
                    matrix_bar_distance_x[i][j] = abs(bar_peak_point[i][0] + bar_peak_point[i][2] -
                                                      bar_peak_point[j][0] - bar_peak_point[j][2]) / 2
        matrix_bar_distance_x[matrix_bar_distance_x == 0] = 1
        # bar pair y distance
        matrix_bar_distance_y = np.zeros(matrix_bar_sum_diff_threshold.shape)
        for i in range(len(matrix_bar_distance_y)):
            for j in range(len(matrix_bar_distance_y)):
                if i < j:
                    matrix_bar_distance_y[i][j] = abs(bar_peak_point[i][1] + bar_peak_point[j][1] -
                                                      bar_peak_point[i][3] - bar_peak_point[j][3]) / 2
        matrix_bar_distance_y[matrix_bar_distance_y == 0] = 0.1  # a trick

        matrix_bar_distance_z = np.zeros(matrix_bar_sum_diff_threshold.shape)
        for i in range(len(matrix_bar_distance_z)):
            for j in range(len(matrix_bar_distance_z)):
                if i < j:
                    matrix_bar_distance_z[i][j] = math.sqrt(pow(bar_peak_point[i][1] - bar_peak_point[j][3], 2) + pow(
                        bar_peak_point[i][0] - bar_peak_point[j][2], 2))
        # FIXME
        matrix_bar_arccos = (np.power(matrix_bar_distance_x, 2) + np.power(matrix_bar_distance_y, 2) -
                             np.power(matrix_bar_distance_z, 2)) / 2 / matrix_bar_distance_x / matrix_bar_distance_y
        matrix_bar_arccos[matrix_bar_arccos > 1] = 1
        matrix_bar_arccos[matrix_bar_arccos < -1] = 1
        matrix_bar_arccos = np.arccos(matrix_bar_arccos) * 180 / math.pi  # transfer measurement into degree
        # 60 < theta < 120
        matrix_bar_arccos_threshold = ((matrix_bar_arccos < 120) & (matrix_bar_arccos > 60)).astype(np.float32)

        # 1.3 < x / y < 3 threshold for the angle
        matrix_bar_ratio = matrix_bar_distance_x / matrix_bar_distance_y
        matrix_bar_ratio_threshold = ((matrix_bar_ratio < 3) & (matrix_bar_ratio > 1.3)).astype(np.float32)
        matrix_match = matrix_bar_ratio_threshold * matrix_bar_sum_diff_threshold * matrix_bar_arccos_threshold
        if self.debug:
            i, j = np.where(matrix_match == 1)
            for k in range(len(i)):
                cv2.circle(mat, (bar_peak_point[i[k]][0], bar_peak_point[i[k]][1]), 3, (0, 255, 255), -1)
                cv2.circle(mat, (bar_peak_point[i[k]][2], bar_peak_point[i[k]][3]), 3, (0, 255, 255), -1)
                cv2.circle(mat, (bar_peak_point[j[k]][0], bar_peak_point[j[k]][1]), 3, (0, 255, 255), -1)
                cv2.circle(mat, (bar_peak_point[j[k]][2], bar_peak_point[j[k]][3]), 3, (0, 255, 255), -1)
                cv2.circle(mat, (int((bar_peak_point[j[k]][2] + bar_peak_point[j[k]][0] + bar_peak_point[i[k]][2] +
                                      bar_peak_point[i[k]][0]) / 4),
                                 int((bar_peak_point[j[k]][3] + bar_peak_point[j[k]][1] + bar_peak_point[i[k]][3] +
                                      bar_peak_point[i[k]][1]) / 4)),
                           5, (255, 255, 0), -1)
            print("matrix_bar_arccos")
            for i in range(len(matrix_bar_arccos)):
                print(matrix_bar_arccos[i])
            print("matrix_bar_arccos_threshold")
            for i in range(len(matrix_bar_arccos_threshold)):
                print(matrix_bar_arccos_threshold[i])
            print("matrix_bar_ratio")
            for i in range(len(matrix_bar_ratio)):
                print(matrix_bar_ratio[i])
            print("matrix_bar_ratio_threshold")
            for i in range(len(matrix_bar_ratio_threshold)):
                print(matrix_bar_ratio_threshold[i])
            print("matrix_bar_sum_diff")
            for i in range(len(matrix_bar_sum_diff)):
                print(matrix_bar_sum_diff[i])
            print("matrix_bar_sum_diff_threshold")
            for i in range(len(matrix_bar_sum_diff_threshold)):
                print(matrix_bar_sum_diff_threshold[i])
            print("matrix_match")
            for i in range(len(matrix_match)):
                print(matrix_match[i])

        if len(np.where(matrix_match == 1)[0]) == 0:
            return None

        # Calculate height sum matrix, sum of pixels
        matrix_bar_pixel_sum = np.zeros(matrix_match.shape)
        i, j = np.where(matrix_match == 1)
        for k in range(len(i)):
            matrix_bar_pixel_sum[i[k]][j[k]] = connect_remain[i[k]][4] + connect_remain[j[k]][4]
        if self.debug:
            print("matrix_bar_height_sum")
            for i in range(len(matrix_bar_pixel_sum)):
                print(matrix_bar_pixel_sum[i])

        # Select Max
        max_i, max_j = np.where(matrix_bar_pixel_sum == np.max(matrix_bar_pixel_sum))
        max_i = max_i[0]
        max_j = max_j[0]
        target_x = int((bar_peak_point[max_j][2] + bar_peak_point[max_j][0] + bar_peak_point[max_i][2] +
                        bar_peak_point[max_i][0]) / 4)
        target_y = int((bar_peak_point[max_j][3] + bar_peak_point[max_j][1] + bar_peak_point[max_i][3] +
                        bar_peak_point[max_i][1]) / 4)
        if self.debug:
            cv2.circle(mat, (target_x, target_y), 8, (100, 100, 250), -1)
            cv2.circle(mat, (bar_peak_point[max_i][2], bar_peak_point[max_i][3]), 3, (100, 100, 0), -1)
            cv2.circle(mat, (bar_peak_point[max_j][2], bar_peak_point[max_j][3]), 3, (100, 100, 0), -1)
            cv2.circle(mat, (bar_peak_point[max_i][0], bar_peak_point[max_i][1]), 3, (100, 100, 0), -1)
            cv2.circle(mat, (bar_peak_point[max_j][0], bar_peak_point[max_j][1]), 3, (100, 100, 0), -1)
            cv2.imshow('Debug', mat)
            set_mouth_callback_show_pix("Debug", mat)

        return target_x, target_y, bar_peak_point[max_j][2]
        # return [target_x, target_y, bar_peak_point[max_j][2]]
        # Why np.array?

    def target_old(self, mat):
        if mat is None:
            return None
        t = None
        self.distance = TargetDis.NEAR
        if self.color == BarColor.BLUE:
            # t = self.track(mat, 80, 2.5, 0.5, 1, 13)
            t = self.track(mat, 55, 2.5, 0.5, 1, 13)
            if t is None:
                t = self.track(mat, 55, 0.8, 0.5, 1, 13)
                # t = self.track(mat, 80, 0.8, 0.5, 1, 13)
        elif self.color == BarColor.RED:
            t = self.track(mat, 50, 2, 0.5, 1, 13)
            if t is None:
                t = self.track(mat, 50, 0.5, 0.3, 0.6, 13)

        if self.debug and t is not None:
            marked = copy.deepcopy(mat)
            x, y, _ = t
            if self.color == BarColor.BLUE:
                line_color = (0, 0, 255)
            else:
                line_color = (255, 0, 0)
            cv2.circle(marked, (x, y), 20, line_color, 2, 15)
            cv2.line(marked, (x - 40, y), (x + 40, y), line_color, 1)
            cv2.line(marked, (x, y - 40), (x, y + 40), line_color, 1)
            cv2.imshow('raw_img', marked)
            set_mouth_callback_show_pix("raw_img", marked)

        if t is not None:
            return t[0], t[1]
        else:
            return None
        # return t[0], t[1] if t is not None else None

    # from datetime import datetime
    # def trace(self, mat):
    #     origin = copy.deepcopy(mat)  # copy image source
    #     t_start = datetime.now()  # get starting time
    #
    #     # track the targets
    #     target = np.array([])
    #     self.distance = TargetDis.NEAR
    #     if self.color == BarColor.BLUE:
    #         target = self.track(mat, 80, 2.5, 0.5, 1, 13)
    #         print("Near: Target: x, y " + str(target))
    #         if len(target) == 0:
    #             self.distance = TargetDis.MID
    #             target = self.track(mat, 80, 0.8, 0.5, 1, 13)
    #             print("Far: Target: x, y " + str(target))
    #     elif self.color == BarColor.RED:
    #         target = self.track(mat, 50, 2, 0.5, 1, 13)
    #         print("Near: Target: x, y " + str(target))
    #         if len(target) == 0:
    #             self.distance = TargetDis.MID
    #             target = self.track(mat, 50, 0.5, 0.3, 0.6, 13)
    #             print("Far: Target: x, y " + str(target))
    #     t_end = datetime.now()  # get ending time
    #     print('running time: ', t_end - t_start)  # print running time
    #     if len(target) > 0:
    #         # MIN_STEP -= 1
    #         # MIN_STEP = 20 if MIN_STEP < 5 else MIN_STEP
    #         self.min_step = 20 if self.min_step <= 5 else self.min_step - 1
    #         # calculate the distance
    #         dist = np.sqrt((np.power(target[0] - self.target_last[0], 2) +
    #                         np.power(target[1] - self.target_last[1], 2)))
    #         if dist > self.min_step:
    #             self.target_last += ((target[:2] - self.target_last) / dist * self.min_step).astype(np.int32)
    #         else:
    #             self.target_last = target[:2]
    #         if self.debug:
    #             cv2.circle(origin, (target[0], target[1]), 8, (0, 255, 0), -1)  # draw a circle on the target
    #     else:
    #         self.min_step = 50 if self.min_step >= 50 else self.min_step
    #         # MIN_STEP += 1
    #         # MIN_STEP = 50 if MIN_STEP > 50 else MIN_STEP
    #
    #     if self.debug:
    #         if self.color == BarColor.BLUE:
    #             cv2.circle(origin, (self.target_last[0], self.target_last[1]), 20, (0, 0, 255), 2, 15)
    #             cv2.line(origin, (self.target_last[0] - 40, self.target_last[1]),
    #                      (self.target_last[0] + 40, self.target_last[1]), (0, 0, 255), 1)
    #             cv2.line(origin, (self.target_last[0], self.target_last[1] - 40),
    #                      (self.target_last[0], self.target_last[1] + 40), (0, 0, 255), 1)
    #         elif self.color == BarColor.RED:
    #             cv2.circle(origin, (self.target_last[0], self.target_last[1]), 20, (255, 0, 0), 2, 15)
    #             cv2.line(origin, (self.target_last[0] - 40, self.target_last[1]),
    #                      (self.target_last[0] + 40, self.target_last[1]), (255, 0, 0), 1)
    #             cv2.line(origin, (self.target_last[0], self.target_last[1] - 40),
    #                      (self.target_last[0], self.target_last[1] + 40), (255, 0, 0), 1)
    #             # out.write(origin)
    #         cv2.imshow('raw_img', origin)
    #         set_mouth_callback_show_pix("raw_img", origin)
    #
    #     return (self.target_last[0], self.target_last[1])
