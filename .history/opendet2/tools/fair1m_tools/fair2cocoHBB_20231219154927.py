from matplotlib import pyplot as plt
import sys

from .fair1m_coco import *

xml_fp = '/content/ISPRS Benchmark/train/labelXml/'
json_path = 'fair1m_coco.json'

fair1m_json(json_path, xml_fp)