# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os
import itertools
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union
import argparse
from collections import defaultdict
import random
import operator
from functools import reduce
from tqdm import tqdm

from detectron2.utils.file_io import PathManager


DIOR_10_CLASS_NAMES = [
    'airplane', 'baseballfield', 'basketballcourt',
    'bridge', 'groundtrackfield', 'harbor', 'ship', 'storagetank',
    'tenniscourt', 'vehicle'
]

DOTA_10_CLASS_NAMES=[
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'harbor'
    ]

DIOR_7_CLASS_NAMES_SET1=[
    'baseballfield', 'bridge', 'groundtrackfield', 'harbor',
    'vehicle', 'tenniscourt', 'storagetank'
]

DOTA_7_CLASS_NAMES_SET1=[
    'baseball-diamond', 'bridge', 'ground-track-field', 'harbor',
    'small-vehicle', 'large-vehicle',   # small vehicle + large vehicle = vehicle 
    'tennis-court', 'storage-tank'
]

DIOR_3_CLASS_NAMES_SET1=[
    'airplane', 'basketballcourt', 'ship'
]

DOTA_3_CLASS_NAMES_SET1=[
    'plane', 'basketball-court', 'ship'
]

DIOR_7_CLASS_NAMES_SET2=[
    'baseballfield', 'bridge', 'groundtrackfield', 'harbor',
    'vehicle', 'basketballcourt', 'airplane'
]

DOTA_7_CLASS_NAMES_SET2=[
    'baseball-diamond', 'bridge', 'ground-track-field', 'harbor',
    'small-vehicle', 'large-vehicle',   # small vehicle + large vehicle = vehicle 
    'basketball-court', 'plane'
]

DIOR_3_CLASS_NAMES_SET2=[
    'storagetank', 'tenniscourt', 'ship'
]

DOTA_3_CLASS_NAMES_SET2=[
    'storage-tank', 'tennis-court', 'ship'
]

DIOR_7_CLASS_NAMES_SET3=[
    'baseballfield', 'bridge', 'groundtrackfield', 'harbor',
    'ship', 'tenniscourt', 'storagetank'
]

DOTA_7_CLASS_NAMES_SET3=[
    'baseball-diamond', 'bridge', 'ground-track-field', 'harbor',
    'ship',   # small vehicle + large vehicle = vehicle 
    'tennis-court', 'storage-tank'
]

DIOR_3_CLASS_NAMES_SET3=[
    'airplane', 'basketballcourt', 'vehicle'
]

DOTA_3_CLASS_NAMES_SET3=[
    'plane', 'basketball-court', 'small-vehicle', 'large-vehicle'
]

DIOR_7_CLASS_NAMES_SET4=[
    'airplane', 'bridge', 'groundtrackfield', 'harbor',
    'ship', 'tenniscourt', 'storagetank'
]

DOTA_7_CLASS_NAMES_SET4=[
    'plane', 'bridge', 'ground-track-field', 'harbor',
    'ship',   # small vehicle + large vehicle = vehicle 
    'tennis-court', 'storage-tank'
]

DIOR_3_CLASS_NAMES_SET4=[
    'baseballfield', 'basketballcourt', 'vehicle'
]

DOTA_3_CLASS_NAMES_SET4=[
    'baseball-diamond', 'basketball-court', 'small-vehicle', 'large-vehicle'
]

DOTA_DIOR_10_CLASS_NAMES_SET1 = tuple(itertools.chain(DIOR_7_CLASS_NAMES_SET1, DIOR_3_CLASS_NAMES_SET1))
DOTA_DIOR_10_CLASS_NAMES_SET2 = tuple(itertools.chain(DIOR_7_CLASS_NAMES_SET2, DIOR_3_CLASS_NAMES_SET2))
DOTA_DIOR_10_CLASS_NAMES_SET3 = tuple(itertools.chain(DIOR_7_CLASS_NAMES_SET3, DIOR_3_CLASS_NAMES_SET3))
DOTA_DIOR_10_CLASS_NAMES_SET4 = tuple(itertools.chain(DIOR_7_CLASS_NAMES_SET4, DIOR_3_CLASS_NAMES_SET4))

def parse_args():
    parser = argparse.ArgumentParser(description='openset voc generator')
    parser.add_argument("--dir", default="datasets/voc_coco", type=str, help="dataset dir")
    parser.add_argument("--in_split", default="DIOR_train", type=str, help="in split name")
    parser.add_argument("--out_split", default="dior_train_openset_cls_spe_0_20", type=str, help="out split name")
    parser.add_argument("--start_class", default="20", type=int)
    parser.add_argument("--end_class", default="40", type=int)
    parser.add_argument("--pre_num_sample", default="8000", type=int)
    parser.add_argument("--post_num_sample", default="5000", type=int)
    return parser.parse_args()

def prepare_openset(dirname: str, in_split: str, out_split: str, start_class: int, end_class: int, pre_num_sample_img: int, post_num_sample_img: int):
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", in_split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=str)

    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))

    image_ids = defaultdict(list)
    for fileid in tqdm(fileids, ncols=80):
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        classes = [obj.find("name").text for obj in tree.findall("object")]
        if (not set(classes).isdisjoint(DOTA_DIOR_10_CLASS_NAMES_SET4[start_class:end_class])) and "person" not in classes and set(classes).isdisjoint(DOTA_DIOR_10_CLASS_NAMES_SET4[end_class:]):
            print(f'pick out classes: {set(classes)}')
            for cls in classes:
                image_ids[cls].append(fileid)
    # count class stastics
    object_counts = {key:len(image_ids[key]) for key in image_ids.keys()}
    total_objects = sum([object_counts[key] for key in object_counts.keys()])
    ratio = float(pre_num_sample_img) / total_objects
    sample_object_counts = {key:int(ratio*object_counts[key]) for key in object_counts.keys()}
    
    sample_image_ids = defaultdict(list)
    for cls in image_ids.keys():
        cls_sample_num = sample_object_counts[cls]
        cls_sample_num = min(cls_sample_num, len(image_ids[cls]))
        sample_image_ids[cls]  = random.sample(image_ids[cls], cls_sample_num)

    # import pdb;pdb.set_trace()
    image_ids = set(reduce(operator.add, [x for _, x in sample_image_ids.items()]))

    post_num_sample_img = min(post_num_sample_img, len(image_ids))
    image_ids = random.sample(image_ids, post_num_sample_img)

    with open(os.path.join(dirname, "ImageSets", "Main", out_split + ".txt"), "w") as f:
        f.writelines("\n".join(image_ids)+"\n")

if __name__ == "__main__":
    args = parse_args()
    prepare_openset(args.dir, args.in_split, args.out_split, args.start_class, args.end_class, args.pre_num_sample, args.post_num_sample)