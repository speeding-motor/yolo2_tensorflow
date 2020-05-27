# -*- coding: utf-8 -*-
# @Time    : 2020-05-22 09:27
# @Author  : speeding_motor

from tensorflow import keras
import tensorflow as tf
from config import GRID_SIZE, BATCH_SIZE, ANCHOR_SIZE, ANCHORS, LOSS_COORD_SCALE, LOSS_NOOBJ_SCALE, LOSS_OBJ_SCALE


class YoloLoss(keras.losses.Loss):

    def __init__(self):
        super(YoloLoss, self).__init__()
        self.priors = tf.reshape(ANCHORS, [1, 1, 1, ANCHOR_SIZE, 2])

        cell_x = tf.reshape(tf.tile(tf.range(GRID_SIZE), [GRID_SIZE]), shape=(1, GRID_SIZE, GRID_SIZE, 1, 1))
        cell_y = tf.transpose(cell_x, perm=(0, 2, 1, 3, 4))
        self.cell_grid = tf.tile(tf.concat([cell_x, cell_y], axis=-1), [BATCH_SIZE, 1, 1, ANCHOR_SIZE, 1])

    def call(self, y_true, y_pred):
        """
        it means there have box when the confidence of box > 0, confidence = IOU

        """
        self.object_mask = tf.cast(y_true[..., 0], dtype=tf.float32)  # whether have boxs or not,[confidence, x, y, w, h]
        self.num_object_box = tf.reduce_sum(self.object_mask)

        self.pred_xy = tf.sigmoid(y_pred[..., 1:3]) + tf.cast(self.cell_grid, tf.float32)
        self.pred_wh = tf.exp(y_pred[..., 3:5]) * self.priors

        coord_loss = self.coordinate_loss(y_true[..., 0:5])
        conf_loss = self.confidence_loss(y_true, y_pred[..., 0])
        classs_loss = self.class_loss(y_true, y_pred)

        return coord_loss + conf_loss + classs_loss

    def coordinate_loss(self, true_boxs):
        """
        coordinate loss contain the xy loss, wh loss
        First: xy_loss ,sigmoid the predict xy, and then add the cell_grad,

        notation: here we just need to compute the loss when the box have object ,if the box don't have box,
        ignore the box coordinate loss

        """
        object_mask = tf.expand_dims(self.object_mask, axis=-1)

        # center_xy loss
        true_center = true_boxs[..., 1:3]
        xy_loss = tf.reduce_sum(tf.square(true_center - self.pred_xy) * object_mask) / (self.num_object_box + 1e-6)

        #  weight & height loss
        true_wh = true_boxs[..., 3:5]
        wh_loss = tf.reduce_sum(tf.square(true_wh - self.pred_wh) * object_mask) / (self.num_object_box + 1e-6)

        return (xy_loss + wh_loss) * LOSS_COORD_SCALE

    def confidence_loss(self, y_true, pred_conf):
        """
        true_conf: = iou between true_box and anchor box, iou(Intersection over union) wrong
        true_conf = iou between true_box and pred_box, and then multiple the probability with object
        conf_mask

        """
        pred_conf = tf.sigmoid(pred_conf)  # adjust pred_conf to 0 ~ 1

        iou = self.iou(y_true[..., 1:5])  # calculate the IOU between true_box and pred_box
        true_conf = iou * y_true[..., 0]

        obj_conf_loss = tf.reduce_sum(tf.square(true_conf - pred_conf) * self.object_mask) / (self.num_object_box + 1e-6)
        noobj_conf_loss = tf.reduce_sum(tf.square(true_conf - pred_conf) * (1 - self.object_mask)) / \
                          (self.num_object_box + 1e-6)

        return obj_conf_loss * LOSS_OBJ_SCALE + noobj_conf_loss * LOSS_NOOBJ_SCALE

    def class_loss(self, y_true, y_pred):
        y_true_class = y_true[..., 5:]
        y_pred_class = y_pred[..., 5:]

        grid_loss = tf.nn.softmax_cross_entropy_with_logits(y_true_class, y_pred_class, axis=-1)
        classs_loss = tf.reduce_sum(grid_loss * self.object_mask) / (self.num_object_box + 1e-6)

        return classs_loss

    def iou(self, true_box):
        """
        :param true_box: shape=[batch_size, grid_h, grid_w, anchor_id ,box], box=[x, y, w, h]
        :param pred_box: shape=[batch_size, grid_h, grid_w, anchor_id ,box], box=[x, y, w, h]
        :return the iou between true_box and pred_box, return shape=[batch_size, grid_h, grid_w, anchor_id, 1]

        """
        # true box radius
        true_box_xy = true_box[..., 0:2]
        true_box_half_wh = true_box[..., 2:4] / 2

        true_box_xy_min = true_box_xy - true_box_half_wh
        true_box_xy_max = true_box_xy + true_box_half_wh

        # pred box radius

        pred_box_half_wh = self.pred_wh / 2

        pred_box_xy_min = self.pred_xy - pred_box_half_wh
        pred_box_xy_max = self.pred_xy + pred_box_half_wh

        inter_section_min_xy = tf.maximum(true_box_xy_min, pred_box_xy_min)
        inter_section_max_xy = tf.minimum(true_box_xy_max, pred_box_xy_max)

        inter_section_wh = tf.maximum(inter_section_max_xy - inter_section_min_xy, 0.)

        union_area = inter_section_wh[..., 0] * inter_section_wh[..., 1]

        pred_area = self.pred_wh[..., 0] * self.pred_wh[..., 1]
        true_area = true_box[..., 2] * true_box[..., 3]

        return union_area / (pred_area + true_area - union_area)


if __name__ == '__main__':
    loss = YoloLoss()