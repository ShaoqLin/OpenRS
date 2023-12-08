import os

from .voc_coco import register_voc_coco
from .dota_dior import (register_dota_dior, register_dota, register_dior7,
                    register_dior7_3, register_dior17, register_dior17_3,
                    register_dior7_3_set1, register_dior7_3_set2, register_dior7_3_set3, 
                    register_dior7_3_set4, register_dior7_set1 ,register_dior7_set2,
                    register_dior7_set3, register_dior7_set4)
from detectron2.data import MetadataCatalog


def register_all_voc_coco(root):
    SPLITS = [
        # VOC_COCO_openset
        ("voc_coco_20_40_test", "voc_coco", "voc_coco_20_40_test"),
        ("voc_coco_20_60_test", "voc_coco", "voc_coco_20_60_test"),
        ("voc_coco_20_80_test", "voc_coco", "voc_coco_20_80_test"),

        ("voc_coco_2500_test", "voc_coco", "voc_coco_2500_test"),
        ("voc_coco_5000_test", "voc_coco", "voc_coco_5000_test"),
        ("voc_coco_10000_test", "voc_coco", "voc_coco_10000_test"),
        ("voc_coco_20000_test", "voc_coco", "voc_coco_20000_test"),

        ("voc_coco_val", "voc_coco", "voc_coco_val"),

    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_voc_coco(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"
        

def register_all_dota_dior(root):
    SPLITS = [ 
        # DOTA_DIOR_openset
        ("dior_10_15_test", "dota_dior", "dior_10_15_test"),
        ("dior_10_20_test", "dota_dior", "dior_10_20_test"),
        
        ("dior_val", "dota_dior", "DIOR_val")
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_dota_dior(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"
        
def register_all_dota_dior_7_3(root):
    SPLITS = [ 
        # DIOR_DOTA_7_3
        # ("DIORval_DOTAtrainval_val", "dota_dior", "DIORval_DOTAtrainval_val"),
        
        ("DIORtest_DOTAtest_7_3_test", "dota_dior", "DIORtest_DOTAtrainval_7_3_test"),
        ("DIORtest_DOTAtest_2500_test", "dota_dior", "DIORtest_DOTAtrainval_2500_test"),
        ("DIORtest_DOTAtest_3500_test", "dota_dior", "DIORtest_DOTAtrainval_3500_test"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_dior7_3(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"
    
def register_all_train_dior7(root):
    SPLITS = [
        # DOTA_closeset
        ("DIOR_train7", "dota_dior", "DIOR_train"),
        ("DIOR_val7", "dota_dior", "DIOR_val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_dior7(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"
        
def register_all_dota_dior_17_3(root):
    SPLITS = [ 
        # DIOR_DOTA_17_3
        # ("DIOR_DOTA_val", "dota_dior", "DIOR_DOTA_val"),
        
        ("DIOR_DOTA_17_20_test", "dota_dior", "DIOR_DOTA_17_20_test"),
        ("DIOR_DOTA_17_20_agn_2500_test", "dota_dior", "DIOR_DOTA_17_20_agn_2500_test"),
        ("DIOR_DOTA_17_20_agn_3000_test", "dota_dior", "DIOR_DOTA_17_20_agn_3000_test"),
        ("DIOR_DOTA_17_20_agn_3500_test", "dota_dior", "DIOR_DOTA_17_20_agn_3500_test"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_dior17_3(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"
    
def register_all_train_dior17(root):
    SPLITS = [
        # DOTA_closeset
        ("DIOR_train17", "dota_dior", "DIOR_train"),
        ("DIOR_val17", "dota_dior", "DIOR_val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_dior17(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"
        
def register_all_dota_dior_7_3_set1(root):
    SPLITS = [ 
        # DIOR_DOTA_7_3
        # ("DIORval_DOTAtrainval_val", "dota_dior", "DIORval_DOTAtrainval_val"),
        
        ("DIORtest_DOTAtrainval_7_3_test_set1", "dota_dior", "DIORtest_DOTAtrainval_7_3_test_set1"),
        ("DOTA_trainval_set1_cls_agn_2000_test", "dota_dior", "DOTA_trainval_set1_cls_agn_2000"),
        ("DOTA_trainval_set1_cls_agn_3000_test", "dota_dior", "DOTA_trainval_set1_cls_agn_3000"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_dior7_3_set1(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"
    
def register_all_train_dior7_set1(root):
    SPLITS = [
        # DOTA_closeset
        ("DIOR_train7_set1", "dota_dior", "DIOR_train"),
        ("DIOR_val7_set1", "dota_dior", "DIOR_val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_dior7_set1(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"

def register_all_dota_dior_7_3_set2(root):
    SPLITS = [ 
        # DIOR_DOTA_7_3
        # ("DIORval_DOTAtrainval_val", "dota_dior", "DIORval_DOTAtrainval_val"),
        
        ("DIORtest_DOTAtrainval_7_3_test_set2", "dota_dior", "DIORtest_DOTAtrainval_7_3_test_set2"),
        ("DOTA_trainval_set2_cls_agn_2000_test", "dota_dior", "DOTA_trainval_set2_cls_agn_2000"),
        ("DOTA_trainval_set2_cls_agn_3000_test", "dota_dior", "DOTA_trainval_set2_cls_agn_3000"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_dior7_3_set2(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"
    
def register_all_train_dior7_set2(root):
    SPLITS = [
        # DOTA_closeset
        ("DIOR_train7_set2", "dota_dior", "DIOR_train"),
        ("DIOR_val7_set2", "dota_dior", "DIOR_val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_dior7_set2(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"

def register_all_dota_dior_7_3_set3(root):
    SPLITS = [ 
        # DIOR_DOTA_7_3
        # ("DIORval_DOTAtrainval_val", "dota_dior", "DIORval_DOTAtrainval_val"),
        
        ("DIORtest_DOTAtrainval_7_3_test_set3", "dota_dior", "DIORtest_DOTAtrainval_7_3_test_set3"),
        ("DOTA_trainval_set3_cls_agn_2000_test", "dota_dior", "DOTA_trainval_set3_cls_agn_2000"),
        ("DOTA_trainval_set3_cls_agn_3000_test", "dota_dior", "DOTA_trainval_set3_cls_agn_3000"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_dior7_3_set3(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"
    
def register_all_train_dior7_set3(root):
    SPLITS = [
        # DOTA_closeset
        ("DIOR_train7_set3", "dota_dior", "DIOR_train"),
        ("DIOR_val7_set3", "dota_dior", "DIOR_val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_dior7_set3(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"

def register_all_dota_dior_7_3_set4(root):
    SPLITS = [ 
        # DIOR_DOTA_7_3
        # ("DIORval_DOTAtrainval_val", "dota_dior", "DIORval_DOTAtrainval_val"),
        
        ("DIORtest_DOTAtrainval_7_3_test_set4", "dota_dior", "DIORtest_DOTAtrainval_7_3_test_set4"),
        ("DOTA_trainval_set4_cls_agn_2000_test", "dota_dior", "DOTA_trainval_set4_cls_agn_2000"),
        ("DOTA_trainval_set4_cls_agn_3000_test", "dota_dior", "DOTA_trainval_set4_cls_agn_3000"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_dior7_3_set4(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"
    
def register_all_train_dior7_set4(root):
    SPLITS = [
        # DOTA_closeset
        ("DIOR_train7_set4", "dota_dior", "DIOR_train"),
        ("DIOR_val7_set4", "dota_dior", "DIOR_val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_dior7_set4(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"
    

if __name__.endswith(".builtin"):
    # Register them all under "./datasets"
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_voc_coco(_root)
    register_all_dota_dior(_root)
    
    register_all_dota_dior_7_3(_root)
    register_all_train_dior7(_root)
    
    register_all_dota_dior_17_3(_root)
    register_all_train_dior17(_root)

    register_all_dota_dior_7_3_set1(_root)
    register_all_train_dior7_set1(_root)
    
    register_all_dota_dior_7_3_set2(_root)
    register_all_train_dior7_set2(_root)
    
    register_all_dota_dior_7_3_set3(_root)
    register_all_train_dior7_set3(_root)
    
    register_all_dota_dior_7_3_set4(_root)
    register_all_train_dior7_set4(_root)
