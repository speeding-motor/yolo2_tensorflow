# -*- coding: utf-8 -*-
# @Time    : 2020-05-22 09:27
# @Author  : speeding_motor
from tensorflow import keras
import tensorflow as tf


class YoloLoss(keras.losses.Loss):
    def __init__(self):
        super(YoloLoss, self).__init__()

    def call(self, y_true, y_pred):
        return y_true - y_pred




