import xml.etree.cElementTree as ET
import os
import itertools
import argparse
from tqdm import tqdm
from pycocotools.coco import COCO
import re

# COCO2VOC_CLASS_NAMES = {
#     "airplane": "aeroplane",
#     "dining table": "diningtable",
#     "motorcycle": "motorbike",
#     "potted plant": "pottedplant",
#     "couch": "sofa",
#     "tv": "tvmonitor",
# }

# DOTA2DIOR_CLASS_NAMES = {
#     "plane": "airplane",
#     "baseball-diamond": "baseballfield",
#     "ground-track-field": "groundtrackfield",
#     "small-vehicle": "vehicle",
#     "large-vehicle": "vehicle",
#     "tennis-court": "tenniscourt",
#     "basketball-court": "basketballcourt",
#     "storage-tank": "storagetank"
# }

FAIR1M2FAIR1M_CLASS_NAMES = {}

DOTA_IGNORE_CLASS_NAME = set([""])

def parse_args():
    parser = argparse.ArgumentParser(description='Convert COCO to VOC style')
    parser.add_argument("--dir", default="/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/FAIR1M1024/train/ann_voc", type=str, help="dataset dir")
    parser.add_argument("--ann_path", default="/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/FAIR1M1024/train/FAIR1m_train1024.json", type=str, help="annotation path")
    return parser.parse_args()

def convert_coco_to_voc(coco_annotation_file, target_folder):
    os.makedirs(os.path.join(target_folder, 'Annotations'), exist_ok=True)
    coco_instance = COCO(coco_annotation_file)
    image_ids = []
    for index, image_id in enumerate(tqdm(coco_instance.imgToAnns, ncols=80)):
        image_details = coco_instance.imgs[image_id]
        annotation_el = ET.Element('annotation')
        ET.SubElement(annotation_el, 'filename').text = image_details['file_name']

        size_el = ET.SubElement(annotation_el, 'size')
        ET.SubElement(size_el, 'width').text = str(image_details['width'])
        ET.SubElement(size_el, 'height').text = str(image_details['height'])
        ET.SubElement(size_el, 'depth').text = str(3)

        for annotation in coco_instance.imgToAnns[image_id]:
            cls_name = coco_instance.cats[annotation['category_id']]['name']
            if not set([cls_name]).isdisjoint(DOTA_IGNORE_CLASS_NAME):
                print(f'ignoring class: {cls_name}')
                continue
            # move here
            object_el = ET.SubElement(annotation_el, 'object')
            # if cls_name in DOTA_IGNORE_CLASS_NAME.keys():
            #     cls_name = DOTA_IGNORE_CLASS_NAME[cls_name]
            ET.SubElement(object_el,'name').text = cls_name
            if len(cls_name) < 1:
                print()
            # ET.SubElement(object_el, 'name').text = 'unknown'
            ET.SubElement(object_el, 'difficult').text = '0'
            bb_el = ET.SubElement(object_el, 'bndbox')
            ET.SubElement(bb_el, 'xmin').text = str(int(annotation['bbox'][0] + 1.0))
            ET.SubElement(bb_el, 'ymin').text = str(int(annotation['bbox'][1] + 1.0))
            ET.SubElement(bb_el, 'xmax').text = str(int(annotation['bbox'][0] + annotation['bbox'][2] + 1.0))
            ET.SubElement(bb_el, 'ymax').text = str(int(annotation['bbox'][1] + annotation['bbox'][3] + 1.0))
        
        file_name = image_details['file_name'].split('.')[0]
        image_ids.append(file_name)
        ET.ElementTree(annotation_el).write(os.path.join(target_folder, 'Annotations', file_name + '.xml'))

    imageset_dir = os.path.join(target_folder, 'ImageSets/Main')
    os.makedirs(imageset_dir, exist_ok=True)
    imageset_name = os.path.basename(coco_annotation_file).split(".json")[0] + ".txt"
    with open(os.path.join(imageset_dir, imageset_name), 'w')  as f:
        f.writelines("\n".join(image_ids)+'\n')
    

if __name__ == '__main__':
    args = parse_args()
    convert_coco_to_voc(args.ann_path, args.dir)
