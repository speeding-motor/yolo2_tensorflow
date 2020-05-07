# -*- coding: utf-8 -*-
# @Time    : 2020-05-01 14:16
# @Author  : speeding_motor

from Model import YoloModel
import pascal
from config import EPOCHS
import numpy as np


def main():
    yolo = YoloModel()

    batch_data = pascal.generate_batch_data()

    for epoch in range(EPOCHS):
        for batch in batch_data.as_numpy_iterator():

            batch_names, batch_boxs = pascal.parse_batch_set(batch)
            train_image = pascal.get_train_images(batch_names)
            train_labels = pascal.get_train_labels(batch_boxs)




if __name__ == '__main__':
    main()