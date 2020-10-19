import numpy as np
import os
import xml.etree.ElementTree as ET
from fvcore.common.file_io import PathManager
import cv2 

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import random
from detectron2.utils.visualizer import Visualizer

__all__ = ["load_voc_hand_instances", "register_pascal_voc"]


# fmt: off
CLASS_NAMES = ["hand"]
# fmt: on


def load_voc_hand_instances(dirname: str, split: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.
    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bndbox = obj.find("bndbox")
            contact_state = obj.find("contact_state").text
            contact_state = contact_state.split(',')[0:4]
            cats = [float(c) for c in contact_state]
 
            bbox = [float(bndbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            px = [float(bndbox.find(x).text) for x in ["x1", "x2", "x3", "x4"]]
            py = [float(bndbox.find(x).text) for x in ["y1", "y2", "y3", "y4"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
   
            poly = [(x, y) for x, y in zip(px, py)]

            instances.append(
                {
                    "category_id": CLASS_NAMES.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly], "cats": cats
                    }
            )

        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_pascal_voc(name, dirname, split, year):
    DatasetCatalog.register(name, lambda: load_voc_hand_instances(dirname, split))
    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES, dirname=dirname, year=year, split=split
    )

splits = ["train", "test", "sub"]
dirname = "./ContactHands/"
for split in splits:
    register_pascal_voc("ContactHands_" + split , dirname, split, 2007)