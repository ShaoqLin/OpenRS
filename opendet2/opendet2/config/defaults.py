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
    _C.INPUT.BOX_NOISE_RATE = 0.0
    
    # config for VOS
    _C.VOS = CN()
    _C.VOS.STARTING_ITER = 12000
    _C.VOS.SAMPLE_NUMBER = 1000
    
    # dataset val
    _C.DATASETS.VAL = None
    
    # Open-set RCNN
    cfg.OPENDET_BENCHMARK = False
    
    cfg.MODEL.PLN = CN()
    cfg.MODEL.PLN.EMD_DIM = 256
    cfg.MODEL.PLN.DISTANCE_TYPE = "COS"  # L1, L2, COS
    cfg.MODEL.PLN.REPS_PER_CLASS = 1 # 5
    cfg.MODEL.PLN.ALPHA = 0.1 # 0.3
    cfg.MODEL.PLN.BETA = 0.9 # 0.7
    cfg.MODEL.PLN.IOU_THRESHOLD = 0.5 # 0.7
    cfg.MODEL.PLN.UNK_THR = 0.4
    cfg.MODEL.PLN.LOSS_WEIGHT = 2.0

    cfg.MODEL.RPN.CTR_REG_LOSS_WEIGHT = 1.0
    cfg.MODEL.RPN.CTR_REG_LOSS_TYPE = "smooth_l1"
    cfg.MODEL.RPN.CTR_SMOOTH_L1_BETA = 0.0
    cfg.MODEL.RPN.IOU_THRESHOLDS_OBJECTNESS = [0.1, 0.3]
    cfg.MODEL.RPN.POSITIVE_FRACTION_OBJECTNESS = 1.0
    cfg.MODEL.RPN.NMS_THRESH_TEST = 1.0

    cfg.MODEL.ROI_BOX_HEAD.IOU_REG_LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_BOX_HEAD.IOU_REG_LOSS_TYPE = "smooth_l1"
    cfg.MODEL.ROI_BOX_HEAD.IOU_SMOOTH_L1_BETA = 0.0
    cfg.MODEL.ROI_BOX_HEAD.CLS_LOSS_WEIGHT = 1.0
    
    cfg.MODEL.ROI_HEADS.MEAN_TYPE = "geometric"
    cfg.MODEL.ROI_HEADS.OBJ_SCORE_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.NUM_KNOWN_CLASSES = 20
    cfg.MODEL.ROI_HEADS.KNOWN_SCORE_THRESH = 0.05
    cfg.MODEL.ROI_HEADS.KNOWN_NMS_THRESH = 0.5
    cfg.MODEL.ROI_HEADS.KNOWN_TOPK = 1000
    cfg.MODEL.ROI_HEADS.UNKNOWN_SCORE_THRESH = 0.05
    cfg.MODEL.ROI_HEADS.UNKNOWN_NMS_THRESH = 0.5
    cfg.MODEL.ROI_HEADS.UNKNOWN_TOPK = 1000
    cfg.MODEL.ROI_HEADS.UNKNOWN_ID= 1000
