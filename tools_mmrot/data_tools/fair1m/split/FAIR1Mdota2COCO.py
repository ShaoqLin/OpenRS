import dota_utils as util
import os
import cv2
import json
from PIL import Image
from tqdm import tqdm

fair1m_names_before = ['Boeing737', 'Boeing777', 'Boeing747', 'Boeing787', 'A320', 'A321', 'A220', 'A330',
                'A350', 'C919', 'ARJ21', 'other-airplane', 'Passenger*Ship', 'Motorboat', 'Fishing*Boat',
                'Tugboat', 'Engineering*Ship', 'Liquid*Cargo*Ship', 'Dry*Cargo*Ship', 'Warship', 'other-ship', 'Small*Car', 'Bus', 'Cargo*Truck',
                'Dump*Truck', 'Van', 'Trailer', 'Tractor', 'Truck*Tractor', 'Excavator', 'other-vehicle', 'Baseball*Field', 'Basketball*Court',
                'Football*Field', 'Tennis*Court', 'Roundabout', 'Intersection', 'Bridge']
fair1m_names_after = ['Boeing737', 'Boeing777', 'Boeing747', 'Boeing787', 'A320', 'A321', 'A220', 'A330',
                'A350', 'C919', 'ARJ21', 'other-airplane', 'passenger ship', 'motorboat', 'fishing boat',
                'tugboat', 'engineering ship', 'liquid cargo ship', 'dry cargo ship', 'warship', 'other-ship', 'small car', 'bus', 'cargo truck',
                'dump truck', 'van', 'trailer', 'tractor', 'truck tractor', 'excavator', 'other-vehicle', 'baseball field', 'basketball court',
                'football field', 'tennis court', 'roundabout', 'intersection', 'bridge']

def DOTA2COCOTrain(srcpath, destfile, cls_names, difficult='2'):
    # set difficult to filter '2', '1', or do not filter, set '-1'

    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'annfiles')

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name.replace('*', ' '), 'supercategory': name.replace('*', ' ')}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(labelparent)
        filenames = tqdm(filenames, ncols=100)
        for file in filenames:
            basename = util.custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + '.tif')
            img = cv2.imread(imagepath)
            height, width, c = img.shape

            single_image = {}
            single_image['file_name'] = basename + '.tif'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            objects = util.parse_dota_poly2(file)
            for obj in objects:
                if obj['difficult'] == difficult:
                    # print('difficult: ', difficult)
                    continue
                single_obj = {}
                single_obj['area'] = obj['area']
                single_obj['category_id'] = cls_names.index(obj['name']) + 1
                obj['name'] = obj['name'].replace('*', ' ')
                single_obj['segmentation'] = []
                single_obj['segmentation'].append(obj['poly'])
                single_obj['iscrowd'] = 0
                xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                         max(obj['poly'][0::2]), max(obj['poly'][1::2])

                width, height = xmax - xmin, ymax - ymin
                single_obj['bbox'] = xmin, ymin, width, height
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
                print(f'file {file.split("/")[-1]} - object {obj["name"]} has beed added')
            image_id = image_id + 1
        json.dump(data_dict, f_out)

def DOTA2COCOTest(srcpath, destfile, cls_names):
    imageparent = os.path.join(srcpath, 'images')
    data_dict = {}

    data_dict['images'] = []
    data_dict['categories'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(imageparent)
        filenames = tqdm(filenames, ncols=100)
        for file in filenames:
            basename = util.custombasename(file)
            imagepath = os.path.join(imageparent, basename + '.jpg')
            img = Image.open(imagepath)
            height = img.height
            width = img.width

            single_image = {}
            single_image['file_name'] = basename + '.jpg'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            image_id = image_id + 1
            # print(f'test file {file.split("/")[-1]} has beed added')
        json.dump(data_dict, f_out)

if __name__ == '__main__':
    # print('converting training dataset from DOTA style to coco style...')
    # DOTA2COCOTrain(r'/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/DOTA1024/train/',
    #                r'/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/DOTA1024/train/DOTA_train1024.json',
    #                wordname_15)
    # DOTA2COCOTrain(r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/trainval1024_ms',
    #                r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/trainval1024_ms/DOTA_trainval1024_ms.json',
    #                wordname_15)
    print('converting testing dataset from DOTA style to coco style...')
    DOTA2COCOTrain(r'/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/FAIR1M1024/val/',
                  r'/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/FAIR1M1024/val/FAIR1m_val1024.json',
                  fair1m_names_before)
    # DOTA2COCOTest(r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/test1024_ms',
    #               r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/test1024_ms/DOTA_test1024_ms.json',
    #               wordname_15)
