# -*- coding: utf-8 -*-
# @Time    : 2020-04-28 10:55
# @Author  : speeding_motor

ANOATATIONS_PATH = "/Users/anyongyi/Downloads/VOC2012/Annotations/"
JPEGS_PATH = "/Users/anyongyi/Downloads/VOC2012/JPEGImages/"

EPOCHS = 100

IMAGE_WIDTH = 416
IMAGE_HEIGHT = 416
IMAGE_CHANNELS = 3

BATCH_SIZE = 12
GRID_SIZE = 13  # cell number per
MAX_BOX_PER_PICTURE = 20
ANCHOR_SIZE = 5
# ANCHORS = [[116, 90], [156, 198], [373, 326], [30, 61], [62, 45], [59, 119], [10, 13], [16, 30], [33, 23]]
ANCHORS = [[10.54, 8.08], [2.43, 2.88], [0.86, 1.10], [3.36, 6.07], [6.38, 6.84]]

PASCAL_VOC_CLASSES = {"person": 1, "bird": 2, "cat": 3, "cow": 4, "dog": 5,
                      "horse": 6, "sheep": 7, "aeroplane": 8, "bicycle": 9,
                      "boat": 10, "bus": 11, "car": 12, "motorbike": 13,
                      "train": 14, "bottle": 15, "chair": 16, "diningtable": 17,
                      "pottedplant": 18, "sofa": 19, "tvmonitor": 20}

CLASS_NUM = len(PASCAL_VOC_CLASSES)
