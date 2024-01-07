from detectron2.data import DatasetCatalog, MetadataCatalog
from .pascal_voc import load_voc_instances

FAIR1M_MAR20_CATEGORIES = [
    # fair1m
    'Boeing737', 'Boeing747', 'Boeing777',
    'Boeing787', 'ARJ21', 'A220', 'A321',
    'A330', 'A350', 'C919',
    # MAR20-10
    'A1', 'A2', 'A3', 'A4', 'A5',
    'A6', 'A7', 'A8', 'A9', 'A10',
    # MAR20-20
    'A11', 'A12', 'A13', 'A14', 'A15',
    'A16', 'A17', 'A18', 'A19', 'A20',
    # Unknown
    "unknown",
]

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
    'A330', 'A350', 'C919',
]

def register_fair1m_mar20(name, dirname, split, year):
    class_names = FAIR1M_MAR20_CATEGORIES
    DatasetCatalog.register(
        name, lambda: load_voc_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )

def register_fair1m(name, dirname, split, year):
    class_names = FAIR1M_PLANE_CATEGORIES
    DatasetCatalog.register(
        name, lambda: load_voc_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )

def register_mar20(name, dirname, split, year):
    class_names = MAR20_CATEGORIES
    DatasetCatalog.register(
        name, lambda: load_voc_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )
