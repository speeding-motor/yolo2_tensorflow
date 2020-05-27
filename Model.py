# -*- coding: utf-8 -*-
# @Time    : 2020-05-01 14:16
# @Author  : speeding_motor

from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Reshape, LeakyReLU, ReLU
import config


class DarkNet19(keras.Model):
    def __init__(self):
        super(DarkNet19, self).__init__()

        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", name='conv_1')
        self.norm1 = BatchNormalization(name='norm1')
        self.relu1 = LeakyReLU(name='relu1')
        self.mpooling1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pooling_1')  # shape = 208 * 208

        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv_2')
        self.norm2 = BatchNormalization(name='norm2')
        self.relu2 = LeakyReLU(name='relu2')
        self.mpooling2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pooling_2')  # shape = 104 * 104

        """first block start"""
        self.conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', name='conv_3')
        self.norm3 = BatchNormalization(name='norm3')
        self.relu3 = LeakyReLU(name='relu3')

        self.conv4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', name='conv_4')
        self.norm4 = BatchNormalization(name='norm4')
        self.relu4 = LeakyReLU(name='relu4')

        self.conv5 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', name='conv_5')
        self.norm5 = BatchNormalization(name='norm5')
        self.relu5 = LeakyReLU(name='relu5')
        self.mpooling3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pooling_3')  # shape = 52 * 52

        self.conv6 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', name='conv_6')
        self.norm6 = BatchNormalization(name='norm6')
        self.relu6 = LeakyReLU(name='relu6')

        self.conv7 = Conv2D(filters=128, kernel_size=(1, 1), padding='same', name='conv_7')
        self.norm7 = BatchNormalization(name='norm7')
        self.relu7 = LeakyReLU(name='relu7')

        self.conv8 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', name='conv_8')
        self.norm8 = BatchNormalization(name='norm8')
        self.relu8 = LeakyReLU(name='relu8')

        self.mpooling4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pooling_4')  # shape = 26 * 26

        """first block end"""

        """second block start"""
        self.conv9 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', name='conv_9')
        self.norm9 = BatchNormalization(name='norm9')
        self.relu9 = LeakyReLU(name='relu9')

        self.conv10 = Conv2D(filters=256, kernel_size=(1, 1), padding='same', name='conv_10')
        self.norm10 = BatchNormalization(name='norm10')
        self.relu10 = LeakyReLU(name='relu10')

        self.conv11 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', name='conv_11')
        self.norm11 = BatchNormalization(name='norm11')
        self.relu11 = LeakyReLU(name='relu11')

        self.conv12 = Conv2D(filters=256, kernel_size=(1, 1), padding='same', name='conv_12')
        self.norm12 = BatchNormalization(name='norm12')
        self.relu12 = LeakyReLU(name='relu12')

        self.conv13 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', name='conv_13')
        self.norm13 = BatchNormalization(name='norm13')
        self.relu13 = LeakyReLU(name='relu13')
        self.mpooling5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pooling_5')  # shape = 13 * 13

        self.conv14 = Conv2D(filters=1024,  kernel_size=(3, 3), padding='same', name='conv_14')
        self.norm14 = BatchNormalization(name='norm14')
        self.relu14 = LeakyReLU(name='relu14')

        self.conv15 = Conv2D(filters=512,  kernel_size=(1, 1), padding='same', name='conv_15')
        self.norm15 = BatchNormalization(name='norm15')
        self.relu15 = LeakyReLU(name='relu15')

        self.conv16 = Conv2D(filters=1024,  kernel_size=(3, 3), padding='same', name='conv_16')
        self.norm16 = BatchNormalization(name='norm16')
        self.relu16 = LeakyReLU(name='relu16')

        self.conv17 = Conv2D(filters=512,  kernel_size=(1, 1), padding='same', name='conv_17')
        self.norm17 = BatchNormalization(name='norm17')
        self.relu17 = LeakyReLU(name='relu17')

        self.conv18 = Conv2D(filters=1024,  kernel_size=(3, 3), padding='same', name='conv_18')
        self.norm18 = BatchNormalization(name='norm18')
        self.relu18 = LeakyReLU(name='relu18')

        self.conv19 = Conv2D(filters=config.ANCHOR_SIZE * (5 + config.CLASS_NUM),  kernel_size=(1, 1), padding='valid',
                             name='conv_19')
        self.relu19 = ReLU(name='relu19')

        self.conv20 = Reshape(target_shape=(config.GRID_SIZE, config.GRID_SIZE, config.ANCHOR_SIZE, 5 + config.CLASS_NUM),
                              name='reshape')

    def call(self, inputs):
        outputs = self.relu1(self.norm1(self.conv1(inputs)))
        outputs = self.mpooling1(outputs)

        outputs = self.relu2(self.norm2(self.conv2(outputs)))
        outputs = self.mpooling2(outputs)

        outputs = self.relu3(self.norm3(self.conv3(outputs)))
        outputs = self.relu4(self.norm4(self.conv4(outputs)))
        outputs = self.relu5(self.norm5(self.conv5(outputs)))
        outputs = self.mpooling3(outputs)

        outputs = self.relu6(self.norm6(self.conv6(outputs)))
        outputs = self.relu7(self.norm7(self.conv7(outputs)))
        outputs = self.relu8(self.norm8(self.conv8(outputs)))
        outputs = self.mpooling4(outputs)

        outputs = self.relu9(self.norm9(self.conv9(outputs)))
        outputs = self.relu10(self.norm10(self.conv10(outputs)))
        outputs = self.relu11(self.norm11(self.conv11(outputs)))
        outputs = self.relu12(self.norm12(self.conv12(outputs)))
        outputs = self.relu13(self.norm13(self.conv13(outputs)))
        outputs = self.mpooling5(outputs)

        outputs = self.relu14(self.norm14(self.conv14(outputs)))
        outputs = self.relu15(self.norm15(self.conv15(outputs)))
        outputs = self.relu16(self.norm16(self.conv16(outputs)))
        outputs = self.relu17(self.norm17(self.conv17(outputs)))
        outputs = self.relu18(self.norm18(self.conv18(outputs)))

        outputs = self.relu19(self.conv19(outputs))
        outputs = self.conv20(outputs)

        return outputs


if __name__ == '__main__':
    model = DarkNet19()