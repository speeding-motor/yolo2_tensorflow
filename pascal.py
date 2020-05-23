# -*- coding: utf-8 -*-
# @Time    : 2020-05-01 14:23
# @Author  : speeding_motor


import tensorflow as tf
import config
from config import JPEGS_PATH, MAX_BOX_PER_PICTURE, GRID_SIZE, ANCHOR_SIZE, CLASS_NUM, ANCHORS
import numpy as np
from util import iou


def generate_batch_data():
    dataset = tf.data.TextLineDataset(filenames="./data/pascal_data.txt")
    dataset = dataset.batch(config.BATCH_SIZE, drop_remainder=False)

    return dataset


def parse_batch_set(batch):
    batch_images = []
    batch_boxs = []
    for line in batch:
        line = bytes.decode(line, 'utf-8')
        line = line.split(',')

        batch_images.append(line[0])
        batch_boxs.append([float(i) for i in line[1:]])

    return batch_images, batch_boxs


def get_train_images(image_names):
    batch_image = []

    for name in image_names:
        image = tf.io.read_file(JPEGS_PATH + name)
        image = tf.io.decode_image(image, channels=config.IMAGE_CHANNELS)
        image = tf.image.resize_with_pad(image, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)

        image = image / 255.0
        batch_image.append(image.numpy())

    return np.array(batch_image)


def get_train_labels(batch_box):
    batch_size = len(batch_box)
    grid_cell_size = config.IMAGE_HEIGHT / config.GRID_SIZE
    """ 
    may have multiple box for per picture, but some picture may have a few box, in order to operation
    need to do :create he each cells label,each cell have ANCHOR_SIZE * (5 +CLASSIFY_NUM)
    1、计算true_box中心x、y位置，根据位置分配label
    2、计算当前true_box与anchor_box的IOU，找出最高IOU的anchor_box计算anchor_id, 并将confidence置为1
    
    can't handle well situation
    1、one grid cell have multiple same object
    2、one grid have more object than the number of anchor box number
    
    """
    batch_boxs_format = np.zeros(shape=(len(batch_box), MAX_BOX_PER_PICTURE * 5))

    for i, picture_boxs in enumerate(batch_box):
        picture_boxs = picture_boxs[0: MAX_BOX_PER_PICTURE * 5]
        batch_boxs_format[i][0: len(picture_boxs)] = picture_boxs

    batch_box = np.reshape(batch_boxs_format, newshape=(batch_size, MAX_BOX_PER_PICTURE, 5))

    batch_xy = (batch_box[..., 3:5] + batch_box[..., 1:3]) // 2
    batch_wh = (batch_box[..., 3:5] - batch_box[..., 1:3])

    batch_grid_id = np.floor(batch_xy / config.IMAGE_HEIGHT * config.GRID_SIZE).astype(int)

    # batch_box[..., 1:3] = (batch_xy - batch_grid_id * grid_cell_size) / grid_cell_size
    batch_box[..., 1:3] = batch_xy / grid_cell_size
    batch_box[..., 3:5] = batch_wh / grid_cell_size

    true_label = np.zeros(shape=(batch_size, GRID_SIZE, GRID_SIZE, ANCHOR_SIZE, 5 + CLASS_NUM))
    vaild_mask = batch_box[..., 0] > 0  # have box mask

    for i in range(batch_size):

        box_wh = batch_wh[i][vaild_mask[i]]
        grid_id = batch_grid_id[i][vaild_mask[i]]
        boxs = batch_box[i][vaild_mask[i]]

        iou_value = iou.IOU().iou(box_wh, anchor_boxs=ANCHORS)

        best_anchor = np.argmax(iou_value, axis=1)  # which anchor is the best anchor for the box

        for j in range(len(iou_value)):

            anchor_id = best_anchor[j]
            class_id = int(boxs[j][0])
            x = grid_id[j][0]
            y = grid_id[j][1]

            true_label[i][x, y, anchor_id][0] = 1.0
            true_label[i][x, y, anchor_id][class_id - 1 + 5] = 1
            true_label[i][x, y, anchor_id][1:5] = boxs[j][1: 5]

    return true_label


if __name__ == '__main__':
    generate_batch_data()