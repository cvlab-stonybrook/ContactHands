import random
import cv2
import os
import argparse
import numpy as np 
import torch
from detectron2.config import get_cfg
from contact_hands_two_stream import CustomVisualizer
from detectron2.data import MetadataCatalog
from contact_hands_two_stream import add_contacthands_config
from datasets import load_voc_hand_instances, register_pascal_voc
from contact_hands_two_stream.engine import CustomPredictor
from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer


class CustomPredictorTwoStream:

    def __init__(self, cfg):
        self.cfg = cfg.clone()  
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image, model2):

        with torch.no_grad():  
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            
            second_stream_outputs = inference_second_stream(model2, original_image)
            predictions = self.model([inputs], second_stream_outputs)[0]
            return predictions

def inference_second_stream(model, image):
    outputs = model(image)   
    return outputs 

def prepare_second_stream():
    cfg2 = get_cfg()
    cfg2.merge_from_file('./configs/second_stream.yaml')
    cfg2.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl"
    model2 = CustomPredictor(cfg2)
    return model2

def prepare_first_stream(cfg_file, weights, roi_score_thresh):
    cfg1 = get_cfg()
    add_contacthands_config(cfg1)
    cfg1.merge_from_file(cfg_file)
    cfg1.MODEL.ROI_HEADS.SCORE_THRESH_TEST = roi_score_thresh
    cfg1.MODEL.WEIGHTS = weights
    model1 = CustomPredictorTwoStream(cfg1)
    
    return model1   

def add_legend(im):
    cyan, magenta, red, yellow = (255, 255, 0), (255, 0, 255), (0, 0, 255),  (0, 255, 255)
    labels = ["No", "Self", "Person", "Object"]
    map_idx_to_color = {}
    map_idx_to_color[0], map_idx_to_color[1], map_idx_to_color[2],  map_idx_to_color[3] = \
    cyan, magenta, red, yellow

    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = im.shape[:2]
    image = 255*np.ones((h+50, w, 3), dtype=np.uint8)
    image[:h, :w, :] = im 
    h, w = image.shape[:2]
    offset = 0

    for itr, word in enumerate(labels):
        offset += int(w / len(labels)) - 50
        cv2.putText(image, word, (offset, h-15), font, 1, map_idx_to_color[itr], 3)

    return image

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments for evaluation')
    
    parser.add_argument('--image_dir', required=True, metavar='path to images', help='path to images')
    parser.add_argument('--ROI_SCORE_THRESH', required=False, metavar='threshold for hand detections', \
    	help='hand detection score threshold', default=0.7)
    parser.add_argument('--sc', required=False, metavar='threshold for self-contact', 
        help='threshold for self-contact', default=0.5)
    parser.add_argument('--pc', required=False, metavar='threshold for person-contact', 
        help='threshold for self-contact', default=0.3)
    parser.add_argument('--oc', required=False, metavar='threshold for object-contact', 
        help='threshold for self-contact', default=0.6)

    args = parser.parse_args()
    images_path = args.image_dir
    roi_score_thresh = float(args.ROI_SCORE_THRESH)
    sc_thresh = float(args.sc)
    pc_thresh = float(args.pc)
    oc_thresh = float(args.oc)
    contact_thresh = [0.5, sc_thresh, pc_thresh, oc_thresh] 
    # if the scores for all contact states is less than corresponding thresholds, No-Contact is predicted; 0.5 is dummy here, it is not used.

    model2 = prepare_second_stream()
    model1 = prepare_first_stream('./configs/ContactHands.yaml', './models/combined_data_model.pth', roi_score_thresh)

    images = sorted(os.listdir(images_path))
    count = 0
    for img in images:
        count += 1
        print(count)
        im = cv2.imread(os.path.join(images_path, img))
        height, width = im.shape[0], im.shape[1]
        ratio = height / width
        im = cv2.resize(im, (720, int(720*ratio)))
        outputs = model1(im, model2)
        v = CustomVisualizer(im[:, :, ::-1], MetadataCatalog.get("ContactHands_test"), scale=1, scores_thresh=contact_thresh)   
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_im = add_legend(v.get_image()[:, :, ::-1])
        cv2.imwrite('./results/res_' + img, out_im)        
