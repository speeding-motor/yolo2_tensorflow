# -*- coding: utf-8 -*-
# @Time    : 2020-04-28 10:58
# @Author  : speeding_motor

import config
import os
import xml.dom.minidom as xdom
import numpy as np
import time


def get_boxs_from_xml(path):
    xml_files = os.listdir(path)

    boxs = []
    for xml in xml_files:
        boxs = parse_xml_for_box(xml, boxs)

    return np.array(boxs)


def parse_xml_for_box(xml, boxs):
    """
    get the box with width, height percent

    """
    dom_tree = xdom.parse(os.path.join(config.ANOATATIONS_PATH, xml))
    sizes = dom_tree.getElementsByTagName('size')
    bndboxs = dom_tree.getElementsByTagName("bndbox")

    width = 0
    height = 0

    cell_size = config.IMAGE_HEIGHT / config.GRID_SIZE

    for size in sizes:
        width = size.getElementsByTagName('width')[0].childNodes[0].data
        height = size.getElementsByTagName('height')[0].childNodes[0].data

    for box in bndboxs:
        x_min = box.getElementsByTagName('xmin')[0].childNodes[0].data
        x_max = box.getElementsByTagName('xmax')[0].childNodes[0].data
        y_min = box.getElementsByTagName('ymin')[0].childNodes[0].data
        y_max = box.getElementsByTagName('ymax')[0].childNodes[0].data

        x_min, x_max, y_min, y_max = reposition_x_y(width, height, x_min, x_max, y_min, y_max)
        relative_w = (float(x_max) - float(x_min)) / cell_size  # get the relative w of the grid cell size
        relative_h = (float(y_max) - float(y_min)) / cell_size

        boxs.append([relative_w, relative_h])
    return boxs


def parse_xml_data(dir):
    xmls = os.listdir(dir)

    images = []
    for xml in xmls:
        image = parse_single_xml_file(xml)
        images.append(image)

    return images


def reposition_x_y(width, height, x_min, x_max, y_min, y_max):
    width, height, x_min, x_max, y_min, y_max = float(width), float(height), float(x_min), float(x_max), float(y_min),\
                                                float(y_max)

    if width <= height:
        scale = height / config.IMAGE_HEIGHT

        pad = (config.IMAGE_WIDTH - width / scale) / 2
        x_min = x_min / scale + pad
        x_max = x_max / scale + pad

        y_min = y_min / scale
        y_max = y_max / scale
    else:
        scale = width / config.IMAGE_WIDTH

        pad = (config.IMAGE_HEIGHT - height / scale) / 2

        x_min = x_min / scale
        x_max = x_max / scale

        y_min = y_min / scale + pad
        y_max = y_max / scale + pad

    return int(x_min), int(x_max), int(y_min), int(y_max)


def parse_single_xml_file(file_name):
    dom_tree = xdom.parse(os.path.join(config.ANOATATIONS_PATH, file_name))
    image_name = dom_tree.getElementsByTagName('filename')[0].childNodes[0].data

    sizes = dom_tree.getElementsByTagName('size')
    objects = dom_tree.getElementsByTagName('object')
    boxs = []

    for size in sizes:
        width = size.getElementsByTagName('width')[0].childNodes[0].data
        height = size.getElementsByTagName('height')[0].childNodes[0].data

    for obj in objects:
        category_name = obj.getElementsByTagName('name')[0].childNodes[0].data
        box = obj.getElementsByTagName('bndbox')[0]

        x_min = box.getElementsByTagName('xmin')[0].childNodes[0].data
        x_max = box.getElementsByTagName('xmax')[0].childNodes[0].data
        y_min = box.getElementsByTagName('ymin')[0].childNodes[0].data
        y_max = box.getElementsByTagName('ymax')[0].childNodes[0].data

        """here we need to calculate the x_min、x_max according to the target image size"""
        x_min, x_max, y_min, y_max = reposition_x_y(width, height, x_min, x_max, y_min, y_max)

        boxs.append([config.PASCAL_VOC_CLASSES[category_name], x_min, y_min, x_max, y_max])

    return [image_name, boxs]


def write_to_txt(data):
    inputs = open('pascal_data.txt', 'a')

    for item in data:
        lines = item[0]
        for box in item[1]:
            lines += "," + str(box[0]) + "," + str(box[1]) + "," + str(box[2]) + "," + str(box[3]) + "," + str(box[4])

        inputs.write(lines + '\n')

    inputs.close()


if __name__ == '__main__':
    """to test the kmeans"""
    # boxs = get_boxs_from_xml(config.ANOATATIONS_PATH)
    # print(boxs.shape)

    dataset = parse_xml_data(config.ANOATATIONS_PATH)
    write_to_txt(dataset)

    print("spend %s " % (time.clock()))

