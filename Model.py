# -*- coding: utf-8 -*-
# @Time    : 2020-05-01 14:16
# @Author  : speeding_motor

from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Reshape
import config


class DarkNet19(keras.Model):
    def __init__(self):
        super(DarkNet19, self).__init__()

        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", name='conv_1')
        self.mpooling1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pooling_1')  # shape = 208 * 208

        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv_2')
        self.mpooling2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pooling_2')  # shape = 104 * 104

        self.conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', name='conv_3')
        self.conv4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', name='conv_4')
        self.conv5 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', name='conv_5')
        self.mpooling3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pooling_3')  # shape = 52 * 52

        self.conv6 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', name='conv_6')
        self.conv7 = Conv2D(filters=128, kernel_size=(1, 1), padding='same', name='conv_7')
        self.conv8 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', name='conv_8')
        self.mpooling4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pooling_4')  # shape = 26 * 26

        self.conv9 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', name='conv_9')
        self.conv10 = Conv2D(filters=256, kernel_size=(1, 1), padding='same', name='conv_10')
        self.conv11 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', name='conv_11')
        self.conv12 = Conv2D(filters=256, kernel_size=(1, 1), padding='same', name='conv_12')
        self.conv13 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', name='conv_13')
        self.mpooling5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pooling_5')  # shape = 13 * 13

        self.conv14 = Conv2D(filters=1024,  kernel_size=(3, 3), padding='same', name='conv_14')
        self.conv15 = Conv2D(filters=512,  kernel_size=(1, 1), padding='same', name='conv_15')
        self.conv16 = Conv2D(filters=1024,  kernel_size=(3, 3), padding='same', name='conv_16')
        self.conv17 = Conv2D(filters=512,  kernel_size=(1, 1), padding='same', name='conv_17')
        self.conv18 = Conv2D(filters=1024,  kernel_size=(3, 3), padding='same', name='conv_18')

        self.conv19 = Conv2D(filters=config.ANCHOR_SIZE * (5 + config.CLASS_NUM),  kernel_size=(1, 1), padding='valid',
                             name='conv_19')
        self.conv20 = Reshape(target_shape=(config.GRID_SIZE, config.GRID_SIZE, config.ANCHOR_SIZE, 5 + config.CLASS_NUM),
                              name='reshape')

    def call(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.mpooling1(outputs)

        outputs = self.conv2(outputs)
        outputs = self.mpooling2(outputs)

        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        outputs = self.mpooling3(outputs)

        outputs = self.conv6(outputs)
        outputs = self.conv7(outputs)
        outputs = self.conv8(outputs)
        outputs = self.mpooling4(outputs)

        outputs = self.conv9(outputs)
        outputs = self.conv10(outputs)
        outputs = self.conv11(outputs)
        outputs = self.conv12(outputs)
        outputs = self.conv13(outputs)
        outputs = self.mpooling5(outputs)

        outputs = self.conv14(outputs)
        outputs = self.conv15(outputs)
        outputs = self.conv16(outputs)
        outputs = self.conv17(outputs)
        outputs = self.conv18(outputs)

        outputs = self.conv19(outputs)
        outputs = self.conv20(outputs)

        return outputs


if __name__ == '__main__':
    model = DarkNet19()