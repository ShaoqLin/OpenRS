from matplotlib import pyplot as plt

import sys
notebook_folders = ['/content/drive/MyDrive/Colab Notebooks/scripts/']

for folder in notebook_folders:
    sys.path.append(folder)

from coco_data.fair1m_coco import *

