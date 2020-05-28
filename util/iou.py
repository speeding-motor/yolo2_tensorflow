# -*- coding: utf-8 -*-
# @Time    : 2020-05-03 10:21
# @Author  : speeding_motor

import numpy as np
import tensorflow as tf


class IOU():
    def __init__(self):
        super(IOU, self).__init__()

    @staticmethod
    def iou_with_anchor(boxs_wh, anchor_boxs):
        """ calculate the iou between the true box and anchor """

        boxs_wh = np.expand_dims(boxs_wh, axis=1)
        anchor_boxs = np.expand_dims(anchor_boxs, axis=0)

        w_min = np.minimum(boxs_wh[..., 0], anchor_boxs[..., 0])
        h_min = np.minimum(boxs_wh[..., 1], anchor_boxs[..., 1])

        intersection_area = w_min * h_min

        boxs_area = boxs_wh[..., 0] * boxs_wh[..., 1]
        anchor_area = anchor_boxs[..., 0] * anchor_boxs[..., 1]

        iou = intersection_area / (boxs_area + anchor_area - intersection_area)

        return iou

    @staticmethod
    def best_iou(true_box, pred_box):
        """
        return the iou between pred_box and true_box
        :param true_box: shape=(true_box_num, 4)
        :param pred_box: shape=(13, 13, anchor_num, 1, 4)
        :return: (13, 13 , anchor_num, true_box_num)
        """
        true_box = tf.cast(true_box, dtype=tf.float32)
        pred_box = tf.cast(pred_box, dtype=tf.float32)

        pred_box = tf.expand_dims(pred_box, axis=-2)

        true_xy = true_box[..., 0:2]
        true_wh = true_box[..., 2:4]
        pred_box_xy = pred_box[..., 0:2]
        pred_box_wh = pred_box[..., 2:4]

        true_xy_min = true_xy - true_wh / 2
        true_xy_max = true_xy + true_wh / 2
        pred_box_xy_min = pred_box_xy - pred_box_wh / 2
        pred_box_xy_max = pred_box_xy + pred_box_wh / 2

        intersection_xy_min = tf.maximum(true_xy_min, pred_box_xy_min)
        intersection_xy_max = tf.minimum(true_xy_max, pred_box_xy_max)
        intersection_wh = tf.maximum(intersection_xy_max - intersection_xy_min, 0)

        pred_area = pred_box[..., 2] * pred_box[..., 3]
        true_area = true_box[..., 2] * true_box[..., 3]

        union_area = intersection_wh[..., 0] * intersection_wh[..., 1]

        return union_area / (pred_area + true_area - union_area)



