# -*- coding: utf-8 -*-
# @Time    : 2020-04-28 12:08
# @Author  : speeding_motor
from data import parse_voc2012_xml
import config
import numpy as np
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot


"""
use the kmeans algorithe according to the iou(intersection over union)
1、calculate the eache sample iou with the clusters
2、choose the most similar box 
3、reset the clusters, and repeat

"""


def kmeans(boxs, k):
    n_samples = boxs.shape[0]
    np.random.seed(42)

    clusters = boxs[np.random.choice(n_samples, k, replace=False)]
    distances = np.zeros(shape=(n_samples, k))

    last_cluster = np.zeros(shape=(n_samples, ))
    ious = []
    i = 0

    while i < 200:
        for row in range(n_samples):
            distances[row] = 1 - iou(boxs[row], clusters)

        nearest_cluster = np.argmin(distances, axis=1)

        if (last_cluster == nearest_cluster).all():
            break

        for i in range(k):
            clusters[i] = np.median(boxs[nearest_cluster == i], axis=0)

        last_cluster = nearest_cluster

        ious.append(np.median(np.median(distances, axis=1)))
        i += 1

    return ious


def iou(box, clusters):
    x_min = np.minimum(box[0], clusters[:, 0])
    y_min = np.minimum(box[1], clusters[:, 1])

    intersection = x_min * y_min

    return intersection / (clusters[:, 0] * clusters[:, 1] + box[0] * box[1] - intersection)


if __name__ == '__main__':

    boxs = parse_voc2012_xml.get_boxs_from_xml(config.ANOATATIONS_PATH)

    box_ious3 = kmeans(boxs=boxs, k=3)
    box_ious5 = kmeans(boxs=boxs, k=5)
    box_ious6 = kmeans(boxs=boxs, k=6)
    box_ious7 = kmeans(boxs=boxs, k=7)
    box_ious9 = kmeans(boxs=boxs, k=9)
    box_ious11 = kmeans(boxs=boxs, k=11)

    pyplot.plot(box_ious3, color='green')
    pyplot.plot(box_ious5, color='black')
    pyplot.plot(box_ious6, color='black')
    pyplot.plot(box_ious7, color='red')
    pyplot.plot(box_ious9, color='blue')
    pyplot.plot(box_ious11, color='blue')
    pyplot.show()
