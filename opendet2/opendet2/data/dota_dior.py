from detectron2.data import DatasetCatalog, MetadataCatalog
from .pascal_voc import load_voc_instances

DOTA_DIOR_CATEGORIES = [
    # DOTA
    'airplane', 'baseballfield', 'bridge',
    'groundtrackfield', 'vehicle', 'ship', 'tenniscourt',
    'basketballcourt', 'storagetank', 'harbor',
    # DIOR-10-15
    'airport', 'chimney', 'dam', 'golffield', 'overpass',
    # DIOR-15-20
    'Expressway-Service-area', 'Expressway-toll-station',
    'stadium', 'trainstation', 'windmill',
    # Unknown
    "unknown",
]

DIOR_CLASS_17_3_NAMES = [
    # DIOR
    'airplane', 'airport', 'baseballfield', 'basketballcourt',
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
    'trainstation', 'vehicle', 'windmill',
    # DIOR-17-20
    'bridge', 'storagetank', 'tenniscourt',
    # Unknown
    "unknown",
]

DIOR_CLASS_17_NAMES = [
    'airplane', 'airport', 'baseballfield', 'basketballcourt',
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
    'trainstation', 'vehicle', 'windmill'
]

DOTA_CATEGORIES = [
    # DOTA
    'airplane', 'baseballfield', 'bridge',
    'groundtrackfield', 'vehicle', 'ship', 'tenniscourt',
    'basketballcourt', 'storagetank', 'harbor',
]

DIOR_7_CATEGORIES = [
    # DIOR
    'baseballfield', 'bridge',
    'groundtrackfield', 'vehicle','tenniscourt',
    'storagetank', 'harbor',
]

DIOR_7_3_CATEGORIES = [
    # DIOR
    'baseballfield', 'bridge',
    'groundtrackfield', 'vehicle','tenniscourt',
    'storagetank', 'harbor',
    # DIOR_DOTA 7+3
    'airplane', 'basketballcourt', 'ship',
    # Unknown
    'unknown',
]

def register_dota_dior(name, dirname, split, year):
    class_names = DOTA_DIOR_CATEGORIES
    DatasetCatalog.register(
        name, lambda: load_voc_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )

def register_dota(name, dirname, split, year):
    class_names = DOTA_CATEGORIES
    DatasetCatalog.register(
        name, lambda: load_voc_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )

def register_dior7(name, dirname, split, year):
    class_names = DIOR_7_CATEGORIES
    DatasetCatalog.register(
        name, lambda: load_voc_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )
    
def register_dior7_3(name, dirname, split, year):
    class_names = DIOR_7_3_CATEGORIES
    DatasetCatalog.register(
        name, lambda: load_voc_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )

def register_dior17_3(name, dirname, split, year):
    class_names = DIOR_CLASS_17_3_NAMES
    DatasetCatalog.register(
        name, lambda: load_voc_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )
    
def register_dior17(name, dirname, split, year):
    class_names = DIOR_CLASS_17_NAMES
    DatasetCatalog.register(
        name, lambda: load_voc_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )