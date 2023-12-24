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


MAR20_CATEGORIES = [
    # MAR20-10
    'A1', 'A2', 'A3', 'A4', 'A5',
    'A6', 'A7', 'A8', 'A9', 'A10',
    # MAR20-20
    'A11', 'A12', 'A13', 'A14', 'A15',
    'A16', 'A17', 'A18', 'A19', 'A20',
]

FAIR1M_PLANE_CATEGORIES = [
    # fair1m
    'Boeing737', 'Boeing747', 'Boeing777',
    'Boeing787', 'ARJ21', 'A220', 'A321',
    'A320', 'A330', 'A350', 'C919', 'other-airplane'
]

DOTA_DIOR_20_CLASS_NAMES = tuple(itertools.chain(FAIR1M_PLANE_CATEGORIES, MAR20_CATEGORIES))

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
        if (not set(classes).isdisjoint(DOTA_DIOR_20_CLASS_NAMES[start_class:end_class])): # and "person" not in classes and set(classes).isdisjoint(DOTA_DIOR_20_CLASS_NAMES[end_class:]):
            print(f'pick out classes: {classes}')
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