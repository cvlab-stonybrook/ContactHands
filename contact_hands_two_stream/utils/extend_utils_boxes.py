from detectron2.structures import Boxes, pairwise_iou
import torch 
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 

def get_union_box(boxes1, boxes2):

    """
    Given two lists of boxes of size N and M,
    compute union boxes 
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.
    Returns:
        boxes3 (Boxes): Contains union boxes obtain 
    """

    boxes1, boxes2 = boxes1.tensor, boxes2.tensor
    if boxes2.shape[0]:
        boxes3 = torch.cat([torch.cat([torch.min(boxes1[i][:2], boxes2[:, :2]), torch.max(boxes1[i][2:], boxes2[:, 2:])], dim=1) for i in range(boxes1.shape[0])], dim=0)
    else:
        device = boxes1.device 
        boxes3 = torch.zeros(0, 4).to(dtype=torch.float32, device=device)  

    return Boxes(boxes3), Boxes(boxes2)

def get_nonzeroiou_unionboxes(boxes1, boxes2):

    iou = pairwise_iou(boxes1, boxes2)
    non_zero = (iou > 0).nonzero()
    union_boxes = []
    for i in range(non_zero.shape[0]):
        pre_union_boxes, _ = get_union_box(Boxes(boxes1.tensor[non_zero[i][0]:non_zero[i][0]+1]), Boxes(boxes2.tensor[non_zero[i][1]:non_zero[i][1]+1]))
        union_boxes.append(pre_union_boxes.tensor)
    if union_boxes:
        union_boxes = torch.cat(union_boxes, dim=0)
        second_boxes = boxes2.tensor[torch.sum(iou>0, dim=0) > 0]
    else:
        device = boxes1.tensor.device 
        union_boxes = torch.zeros(0, 4).to(dtype=torch.float32, device=device)
        second_boxes = torch.zeros(0, 4).to(dtype=torch.float32, device=device)

    return Boxes(union_boxes), Boxes(second_boxes)  