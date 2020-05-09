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

finally output the cluster w, h

"""


def kmeans(boxs_wh, k):
    """
    here we need to calculate the new centroids of the each cluster
    1、random the k example as the k_cluster, and then calculate the iou for between k_cluster and each sample
    2、get the best iou of for each example, and assign the box to the k_cluster,
    3、sum the each k_cluster width 、height, and divide the number of k_cluster samples,
    4、use the new width 、height as the new cluster centroids, and repeat 1、2、3、4

    """
    n_samples = boxs_wh.shape[0]
    sample_dim = boxs_wh.shape[1]
    np.random.seed(42)

    clusters = boxs_wh[np.random.choice(n_samples, k, replace=False)]
    distances = np.zeros(shape=(n_samples, k))

    last_cluster = np.zeros(shape=(n_samples, ))
    ious = []

    while True:
        for row in range(n_samples):
            distances[row] = 1 - iou(boxs_wh[row], clusters)

        nearest_cluster = np.argmin(distances, axis=1)

        if (last_cluster == nearest_cluster).all():
            write_culuster(clusters, k)
            break

        # for i in range(k):
        #     clusters[i] = np.median(boxs[nearest_cluster == i], axis=0)
        #
        # last_cluster = nearest_cluster
        #
        # ious.append(np.median(np.median(distances, axis=1)))
        wh_sum = np.zeros(shape=(k, sample_dim), dtype=float)
        for i in range(n_samples):
            wh_sum[nearest_cluster[i]] += boxs_wh[i]

        for i in range(k):
            clusters[i] = wh_sum[i] / np.sum(nearest_cluster == i)

        last_cluster = nearest_cluster

        iou_means = np.sum(distances) / (n_samples * k)
        ious.append(iou_means)
    return ious


def write_culuster(clusters, k):
    with open('./data/anchor_box.txt1', 'w') as f:
        for item in clusters:
            line = ""
            for i in range(len(item)):
                line += str(item[i]) + " "
            line += "\n"
            f.write(line)
        f.close()


def iou(box, clusters):

    x_min = np.minimum(box[0], clusters[:, 0])
    y_min = np.minimum(box[1], clusters[:, 1])

    intersection = x_min * y_min

    return intersection / (clusters[:, 0] * clusters[:, 1] + box[0] * box[1] - intersection)


if __name__ == '__main__':

    boxs_wh = parse_voc2012_xml.get_boxs_from_xml(config.ANOATATIONS_PATH)

    box_ious3 = kmeans(boxs_wh=boxs_wh, k=3)
    box_ious5 = kmeans(boxs_wh=boxs_wh, k=5)
    # box_ious6 = kmeans(boxs_wh=boxs_wh, k=6)
    # box_ious7 = kmeans(boxs_wh=boxs_wh, k=7)
    box_ious9 = kmeans(boxs_wh=boxs_wh, k=9)
    # box_ious11 = kmeans(boxs_wh=boxs_wh, k=11)
    #
    pyplot.plot(box_ious3, color='green')
    pyplot.plot(box_ious5, color='black')
    # pyplot.plot(box_ious6, color='black')
    # pyplot.plot(box_ious7, color='red')
    pyplot.plot(box_ious9, color='blue')
    # pyplot.plot(box_ious11, color='blue')
    pyplot.show()
