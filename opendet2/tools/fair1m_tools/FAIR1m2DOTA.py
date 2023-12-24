from xml.etree import ElementTree as ET
import os
from tqdm import tqdm

items = 'Boeing737, Boeing777, Boeing747, Boeing787, AirbusA320, AirbusA321, AirbusA220, Airbus A330, \
        Airbus A350, COMAC C919, COMAC ARJ21, other-airplane, passenger ship, motorboat, fishing boat, \
        tugboat, engineering ship, liquid cargo ship, dry cargo ship, warship, other-ship, small car, bus, cargo truck, \
        dump truck, van, trailer, tractor, truck tractor, excavator, other-vehicle, baseball field, basketball court, \
        football field, tennis court, roundabout, intersection, bridge'
items = [item.strip() for item in items.split(',')]
convert_options = {}

# for item in items:
#     convert_options[item] = item.replace(' ','-')
#     print(f'convert item {item} to {item.replace(" ","-")}')

    # if 'Boeing' in item or 'Airbus' in item or 'COMAC' in item or item == 'other-airplane':
    #     if 'Airbus' in item:
    #         convert_options[item.replace('Airbus ', 'Airbus').lower()] = 'plane'
    #     elif 'COMAC' in item:
    #         convert_options[item.replace('COMAC ', 'COMAC').lower()] = 'plane'
    #     else:
    #         convert_options[item] = 'plane'
    #         convert_options[item.lower()] = 'plane'
    #         convert_options[item.lower().replace(' ','')] = 'plane'
    # elif item in ['small car', 'bus', 'cargo truck', 'dump truck', 'van', 'trailer', 'tractor', 'truck tractor', 'other-vehicle', 'excavator']:
    #     convert_options[item] = 'vehicle'
    # else:
    #     print ("Skipping ", item)
    #     convert_options[item] = 'ignore'

def convert_XML_to_DOTA(filename):
    mydoc = ET.parse(filename)
    root = mydoc.getroot()

    objects = root.find('objects')
    
    items = objects.findall('object')
    output_file = os.path.splitext(os.path.split(filename)[-1])[0] + '.txt'
    with open(f'/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/FAIR1M2.0/validation/labelTxt/fair1m_val_{output_file}', 'w') as f:
        ann_list = []
        for item in items:
            label = item.find('possibleresult')
            points = item.find('points')
            label=label.find('name').text
            mapped_label = label.replace(' ', '*')
            if mapped_label != 'ignore':
                points = [[int(float(item)) for item in point.text.split(',')] for point in points.findall('point')]
                x1, y1 = points[0]
                x2, y2 = points[1]
                x3, y3 = points[2]
                x4, y4 = points[3]
                ann = [x1, y1, x2, y2, x3, y3, x4, y4, mapped_label, 1]
                ann = [str(item) for item in ann]
                ann_list.append(' '.join(ann))
                print(label, mapped_label, x1, y1, x2, y2, x3, y3, x4, y4)

        f.write('\n'.join(ann_list))

if __name__ ==  '__main__':
    xml_files = os.listdir('/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/FAIR1M2.0/validation/labelXmls/labelXml')
    os.makedirs('/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/FAIR1M2.0/validation/labelTxt', exist_ok=True)
    for file in tqdm(xml_files, ncols=80):
        convert_XML_to_DOTA(os.path.join('/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/FAIR1M2.0/validation/labelXmls/labelXml', file))
