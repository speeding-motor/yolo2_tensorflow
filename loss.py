# -*- coding: utf-8 -*-
# @Time    : 2020-05-22 09:27
# @Author  : speeding_motor

from tensorflow import keras
import tensorflow as tf
from config import GRID_SIZE, BATCH_SIZE, ANCHOR_SIZE


class YoloLoss(keras.losses.Loss):

    def __init__(self):
        super(YoloLoss, self).__init__()

        cell_x = tf.reshape(tf.tile(tf.range(GRID_SIZE), [GRID_SIZE]), shape=(1, GRID_SIZE, GRID_SIZE, 1, 1))
        cell_y = tf.transpose(cell_x, perm=(0, 2, 1, 3, 4))

        self.cell_grid = tf.tile(tf.concat([cell_x, cell_y], axis=-1), [BATCH_SIZE, 1, 1, ANCHOR_SIZE, 1])

    def call(self, y_true, y_pred):
        boxs_mask = tf.cast(tf.expand_dims(y_true[..., 0], axis=-1), dtype=tf.float32)  # whether have boxs or not,[confidence, x, y, w, h]

        coord_loss = self.coordinate_loss(y_true[..., 1:3], y_pred[..., 1:3], boxs_mask)
        wh_loss = self.coordinate_



        return coord_loss

    def coordinate_loss(self, true_xy, pred_xy, boxs_mask):
        """
        coordinate loss contain the xy loss, wh loss
        First: xy_loss ,sigmoid the predict xy, and then add the cell_grad,

        notation: here we just need to compute the loss when the box have object ,if the box don't have box,
        ignore the box coordinate loss

        """
        nb_boxs = tf.reduce_sum(tf.cast(boxs_mask, dtype=tf.float32))

        pred_coord_xy = tf.sigmoid(pred_xy) + tf.cast(self.cell_grid, dtype=tf.float32)

        coord_loss = tf.reduce_sum(tf.square(true_xy - pred_coord_xy) * boxs_mask) / (nb_boxs + 1e-6)

        return coord_loss


if __name__ == '__main__':
    loss = YoloLoss()
    loss([1], [0])
