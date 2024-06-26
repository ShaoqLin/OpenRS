'''
Author: ShaoqLin 852884179@qq.com
Date: 2023-11-10 21:06:16
LastEditors: ShaoqLin 852884179@qq.com
LastEditTime: 2023-11-26 09:39:27
FilePath: /OpenRS/opendet2/opendet2/config/defaults.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from detectron2.config import CfgNode as CN


def add_opendet_config(cfg):
    _C = cfg

    # unknown probability loss
    _C.UPLOSS = CN()
    _C.UPLOSS.START_ITER = 100  # usually the same as warmup iter
    _C.UPLOSS.SAMPLING_METRIC = "min_score"
    _C.UPLOSS.TOPK = 3
    _C.UPLOSS.ALPHA = 1.0
    _C.UPLOSS.WEIGHT = 0.5

    # instance contrastive loss
    _C.ICLOSS = CN()
    _C.ICLOSS.OUT_DIM = 128
    _C.ICLOSS.QUEUE_SIZE = 256
    _C.ICLOSS.IN_QUEUE_SIZE = 16
    _C.ICLOSS.BATCH_IOU_THRESH = 0.5
    _C.ICLOSS.QUEUE_IOU_THRESH = 0.7
    _C.ICLOSS.TEMPERATURE = 0.1
    _C.ICLOSS.WEIGHT = 0.1

    # register RoI output layer
    _C.MODEL.ROI_BOX_HEAD.OUTPUT_LAYERS = "FastRCNNOutputLayers"
    # known classes
    _C.MODEL.ROI_HEADS.NUM_KNOWN_CLASSES = 20
    _C.MODEL.RETINANET.NUM_KNOWN_CLASSES = 20
    # thresh for visualization results.
    _C.MODEL.ROI_HEADS.VIS_IOU_THRESH = 1.0
    # scale for cosine classifier
    _C.MODEL.ROI_HEADS.COSINE_SCALE = 20

    # swin transformer
    _C.MODEL.SWINT = CN()
    _C.MODEL.SWINT.EMBED_DIM = 96
    _C.MODEL.SWINT.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
    _C.MODEL.SWINT.DEPTHS = [2, 2, 6, 2]
    _C.MODEL.SWINT.NUM_HEADS = [3, 6, 12, 24]
    _C.MODEL.SWINT.WINDOW_SIZE = 7
    _C.MODEL.SWINT.MLP_RATIO = 4
    _C.MODEL.SWINT.DROP_PATH_RATE = 0.2
    _C.MODEL.SWINT.APE = False
    _C.MODEL.BACKBONE.FREEZE_AT = -1
    _C.MODEL.FPN.TOP_LEVELS = 2

    # solver, e.g., adamw for swin
    _C.SOLVER.OPTIMIZER = 'SGD'
    _C.SOLVER.BETAS = (0.9, 0.999)

    # config for siren
    _C.SIREN = CN()
    _C.SIREN.LOSS_WEIGHT = 0.1
    _C.SIREN.PROJECTION_DIM = 32   

    # config to data noise
    _C.DATASETS.AUG = CN()
    _C.DATASETS.AUG.NAME = ""
    _C.DATASETS.AUG.AUG_RATE = 0.0
    