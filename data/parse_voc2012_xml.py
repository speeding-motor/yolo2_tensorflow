# -*- coding: utf-8 -*-
# @Time    : 2020-04-28 10:58
# @Author  : speeding_motor

import glob
import config
import os
import xml.dom.minidom as xdom
import numpy as np


def get_boxs_from_xml(path):
    xml_files = os.listdir(path)

    boxs = []
    for xml in xml_files:
        boxs = parse_xml_file(xml, boxs)

    return np.array(boxs)


def parse_xml_file(xml, boxs):
    """
    get the box with width, height percent

    """
    dom_tree = xdom.parse(os.path.join(config.ANOATATIONS_PATH, xml))
    sizes = dom_tree.getElementsByTagName('size')
    bndboxs = dom_tree.getElementsByTagName("bndbox")

    width = 0
    height = 0

    for size in sizes:
        width = size.getElementsByTagName('width')[0].childNodes[0].data
        height = size.getElementsByTagName('height')[0].childNodes[0].data

    for box in bndboxs:
        xmin = box.getElementsByTagName('xmin')[0].childNodes[0].data
        xmax = box.getElementsByTagName('xmax')[0].childNodes[0].data
        ymin = box.getElementsByTagName('ymin')[0].childNodes[0].data
        ymax = box.getElementsByTagName('ymax')[0].childNodes[0].data

        w = (float(xmax) - float(xmin)) / float(width)
        h = (float(ymax) - float(ymin)) / float(height)

        boxs.append([w, h])
    return boxs


if __name__ == '__main__':
    boxs = get_boxs_from_xml(config.ANOATATIONS_PATH)
    print(boxs.shape)