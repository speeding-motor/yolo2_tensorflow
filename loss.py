# -*- coding: utf-8 -*-
# @Time    : 2020-05-22 09:27
# @Author  : speeding_motor

from tensorflow import keras
import tensorflow as tf
from config import GRID_SIZE, BATCH_SIZE, ANCHOR_SIZE, ANCHORS, LAMBDA_COORD, LAMBDA_NOOBJ, LAMBDA_OBJ \
    , THRESHOLD_IOU
from util.iou import IOU


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
        object_mask = tf.cast(y_true[..., 0], dtype=tf.float32)  # whether have box or not[confidence, x, y, w, h]
        object_mask_bool = tf.cast(object_mask, dtype=bool)
        num_object_mask = tf.reduce_sum(object_mask)

        pred_xy = tf.sigmoid(y_pred[..., 1:3]) + tf.cast(self.cell_grid, tf.float32)
        pred_wh = tf.exp(y_pred[..., 3:5]) * self.priors

        def coordinate_loss(true_boxs):
            """
            coordinate loss contain the xy loss, wh loss
            First: xy_loss ,sigmoid the predict xy, and then add the cell_grad,
            notation: here we just need to compute the loss when the box have object ,if the box don't have box,
            ignore the box coordinate loss
            """
            object_mask_expand = tf.expand_dims(object_mask, axis=-1)  # shape = [batch_size, h, w, anchor_num, 1]

            true_xy = true_boxs[..., 1:3]
            xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask_expand)

            true_wh = true_boxs[..., 3:5]
            wh_loss = tf.reduce_sum(tf.square(true_wh - pred_wh) * object_mask_expand)

            return (xy_loss + wh_loss) * LAMBDA_COORD

        def confidence_loss():
            """
            true_conf: = iou between true_box and anchor box, iou(Intersection over union) wrong
            true_conf = iou between true_box and pred_box, and then multiple the probability with object
            conf_mask

            """
            pred_conf = tf.sigmoid(y_pred[..., 0])  # adjust pred_conf to 0 ~ 1, shape = [batch_size, h, w, anchor_num]
            pred_box = tf.concat([pred_xy, pred_wh], axis=-1)

            # calculate the IOU between true_box and pred_box
            # true_conf = IOU.iou(y_true[..., 1:5], pred_box) * y_true[..., 0]
            ignore_mask = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)

            def loop_body(b, ignore_mask):
                """ get get iou between ground truth and pred_box """
                true_box = tf.boolean_mask(y_true[b][..., 1: 5], object_mask_bool[b])  # shape = ()

                true_box = tf.reshape(true_box, [1, 1, 1, -1, 4])
                iou_scores = IOU.best_iou(true_box, pred_box[b])  # return the shape [13, 13, 5, len(true_box)]
                best_ious = tf.reduce_max(iou_scores, axis=-1)
                ignore_mask = ignore_mask.write(b, tf.cast(best_ious < THRESHOLD_IOU, dtype=tf.float32))

                best_ious_debug = tf.boolean_mask(best_ious, best_ious > THRESHOLD_IOU)

                return b + 1, ignore_mask

            _, ignore_mask = tf.while_loop(lambda b, *args: b < BATCH_SIZE, loop_body, [0, ignore_mask])
            ignore_mask = ignore_mask.stack()

            obj_conf_loss = tf.reduce_sum(tf.square(1 - pred_conf) * object_mask)
            noobj_conf_loss = tf.reduce_sum(tf.square(- pred_conf) * (1 - object_mask) * ignore_mask)

            return obj_conf_loss * LAMBDA_OBJ + noobj_conf_loss * LAMBDA_NOOBJ

        def class_loss():
            true_class = y_true[..., 5:]
            pred_class = y_pred[..., 5:]

            loss_cell = tf.nn.softmax_cross_entropy_with_logits(true_class, pred_class, axis=-1)
            c_loss = tf.reduce_sum(loss_cell * object_mask)

            return c_loss

        # coord_loss = coordinate_loss(y_true[..., 0:5])
        conf_loss = confidence_loss()
        # classs_loss = class_loss()

        # return (coord_loss + conf_loss + classs_loss) / BATCH_SIZE
        return conf_loss / BATCH_SIZE

if __name__ == '__main__':
    loss = YoloLoss()