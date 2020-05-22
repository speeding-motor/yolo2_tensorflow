# -*- coding: utf-8 -*-
# @Time    : 2020-05-01 14:16
# @Author  : speeding_motor

from Model import DarkNet19
from loss import YoloLoss
import pascal
from config import EPOCHS
import numpy as np
import tensorflow as tf


def main():
    yolo = DarkNet19()

    batch_data = pascal.generate_batch_data()

    yololoss = YoloLoss()
    optimizer = tf.optimizers.Adam()

    for epoch in range(EPOCHS):
        for batch in batch_data.as_numpy_iterator():

            batch_names, batch_boxs = pascal.parse_batch_set(batch)
            train_image = pascal.get_train_images(batch_names)
            train_labels = pascal.get_train_labels(batch_boxs)

            with tf.GradientTape() as tape:
                y_pred = yolo(train_image)
                loss = yololoss(y_true=train_labels, y_pred=y_pred)

            gradient = tape.gradient(loss, yolo.trainable_variables)
            optimizer.apply_gradients(grads=gradient)

        print(loss)


if __name__ == '__main__':
    main()