# -*- coding: utf-8 -*-
# @Time    : 2020-05-03 10:21
# @Author  : speeding_motor

import numpy as np


class IOU():
    def __init__(self):
        super(IOU, self).__init__()

    def iou(self, boxs, anchor_boxs):
        """ calculate the iou between the true box and anchor """
        boxs_wd = boxs[..., 3: 5] - boxs[..., 1: 3]

        boxs_wd = np.expand_dims(boxs_wd, axis=1)
        anchor_boxs = np.expand_dims(anchor_boxs, axis=0)

        w_min = np.minimum(boxs_wd[..., 0], anchor_boxs[..., 0])
        h_min = np.minimum(boxs_wd[..., 1], anchor_boxs[..., 1])

        intersection_area = w_min * h_min

        boxs_area = boxs_wd[..., 0] * boxs_wd[..., 1]
        anchor_area = anchor_boxs[..., 0] * anchor_boxs[..., 1]

        iou = intersection_area / (boxs_area + anchor_area - intersection_area)

        return iou