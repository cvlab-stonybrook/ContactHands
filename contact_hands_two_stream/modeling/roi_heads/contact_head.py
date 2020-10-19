import torch
from torch import nn
from torch.nn import functional as F 
from detectron2.layers import Conv2d, ShapeSpec, cat, get_norm 
from detectron2.utils.registry import Registry
import numpy as np 
import fvcore.nn.weight_init as weight_init
from .cross_fetaure_affinity_pooling import CrossFeatureAffinityPooling
from .spatial_attention import SpatialAttention

ROI_CONTACT_HEAD_REGISTRY = Registry("ROI_CONTACT_HEAD")
ROI_CONTACT_HEAD_REGISTRY.__doc__ == """
Registry for contact heads, which make contact predictions from per-region features.

The registered object will be called with obj(cfg,, input_shape).
"""

def contact_loss(pred_raw_scores, instances, pos_weight, device):
    """
    Compute the binary cross-entropy loss (Multi-label class loss)

    Args:
        pred_raw_scores (Tensor): A tensor of shape (B, num_cats), where B is the 
            total number of predicted contacts in all the images, num_cats is the total
            number of possible contact states.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            cats) associated with each instance are stored in fields.
    
    Returns:
        contact_loss (Tensor): A scalar tensor containing the loss.
    """
    gt_cats = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        gt_cats.append(instances_per_image.gt_cats)
    
    if len(gt_cats) == 0:
        return pred_raw_scores.sum() * 0
    
    gt_cats = cat(gt_cats, dim=0)
    
    # Do not consider categories marked "unsure (marked by 2)"
    row_mask = (gt_cats < 2).sum(dim=1) == 4
    gt_cats = gt_cats[row_mask, :]
    pred_raw_scores = pred_raw_scores[row_mask, :]
    
    if not gt_cats.shape[0]:
        return pred_raw_scores.sum() * 0
    
    if pos_weight:
        pos_weight = torch.FloatTensor(pos_weight).to(device)
    else:
        pos_weight = None 

    contact_loss = F.binary_cross_entropy_with_logits(pred_raw_scores, gt_cats, pos_weight=pos_weight)
    return contact_loss

def contact_head_inference(pred_raw_scores, pred_instances):
    """
    Convert the raw scores of the contact head to sigmoid scores and add new
    "pred_cats" field to pred_instances.
    
    Args:
        pred_raw_scores (Tensor): A tensor of shape (B, num_cats), where B is the
            total number of predicted contact states in all the images, num_cats
            is the total number of possible contact states.
        pred_instances (list[Instances]): A list of N Instances, where N is the
            number of images in the batch.  

    Returns:
        None. pred_instances will contain an extra "pred_cats" field storing a Tensor 
            of shape (num_cats) for predicted class
    """
    pred_cats = pred_raw_scores.sigmoid()
    num_boxes_per_image = [len(i) for i in pred_instances]
    pred_cats = pred_cats.split(num_boxes_per_image, dim=0)

    for cats, instances in zip(pred_cats, pred_instances):
        instances.pred_cats = cats 

@ROI_CONTACT_HEAD_REGISTRY.register()
class ContactHead(nn.Module):
    """
    A head with several fc layers (each followed by relu if there is more than one FC)
    """
    def __init__(self, cfg, input_shape: ShapeSpec):
        
        super(ContactHead, self).__init__()

        hand_fcs = cfg.MODEL.ROI_CONTACT_HEAD.HAND_FCS
        hand_object_fcs = cfg.MODEL.ROI_CONTACT_HEAD.HAND_OBJECT_FCS
        cross_attention_fcs = cfg.MODEL.ROI_CONTACT_HEAD.CROSS_ATTENTION_FCS
        project_dims = cfg.MODEL.ROI_CONTACT_HEAD.PROJECT_DIMS
        num_spatial_attns = cfg.MODEL.ROI_CONTACT_HEAD.NUM_SPATIAL_ATTENTION
        self.pos_weight = cfg.MODEL.ROI_CONTACT_HEAD.POS_WEIGHT 
        self.device = cfg.MODEL.DEVICE 

        assert len(hand_fcs) > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        self.hand_fcs = []
        for k, fc_dim in enumerate(hand_fcs):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc_hand_{}".format(k + 1), fc)
            self.hand_fcs.append(fc)
            self._output_size = fc_dim
        for layer in self.hand_fcs:
            weight_init.c2_xavier_fill(layer)

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        self.hand_object_fcs = []
        for k, fc_dim in enumerate(hand_object_fcs):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc_hand_object_{}".format(k + 1), fc)
            self.hand_object_fcs.append(fc)
            self._output_size = fc_dim
        for layer in self.hand_object_fcs:
            weight_init.c2_xavier_fill(layer)

        self.cross_attention = CrossFeatureAffinityPooling(input_shape.channels)
        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        self.cross_attention_fcs = []
        for k, fc_dim in enumerate(cross_attention_fcs):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("cross_attention_fc{}".format(k + 1), fc)
            self.cross_attention_fcs.append(fc)
            self._output_size = fc_dim   
        for layer in self.cross_attention_fcs:
            weight_init.c2_xavier_fill(layer)

        self.spatial_attention = SpatialAttention(input_shape.channels, num_spatial_attns)


        self.project_fcs = []
        for k, fc_dim in enumerate(project_dims):
            if k==0:
                fc = nn.Linear(np.prod(self._output_size*3), fc_dim)
            else:
                fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("project_fc{}".format(k + 1), fc)
            self.project_fcs.append(fc)
            self._output_size = fc_dim   

        for layer in self.project_fcs:
            weight_init.c2_xavier_fill(layer)

        self.classifier =nn. Linear(self._output_size, 4)
        weight_init.c2_xavier_fill(self.classifier)

    def forward(self, hand_features, hand_object_features, instances):
        """
        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        K_cross_attention_hands = []
        num_hands = hand_features.shape[0]
        for i in range(num_hands):
            if hand_object_features[i].shape[0]:
                hand_ftr = hand_features[i:i+1].repeat(hand_object_features[i].shape[0], 1, 1, 1) # [M_i, 7, 7, 256]
                hand_obj_ftr = hand_object_features[i] #[M_i, 7, 7, 256]
                cross_attn_ftrs = self.cross_attention(hand_ftr, hand_obj_ftr) #[M_i, 7, 7, 256]
                K_cross_attention_hands.append(cross_attn_ftrs)
            else:
                hand_ftr = hand_features[i:i+1]
                K_cross_attention_hands.append(hand_ftr)

        K_cross_attention_hands_fcs = []
        for feature in K_cross_attention_hands:
            feature = torch.flatten(feature, start_dim=1) #[M_i, -1]
            for layer in self.cross_attention_fcs:
                feature = F.relu(layer(feature))
            K_cross_attention_hands_fcs.append(feature)


        hand_features = torch.flatten(hand_features, start_dim=1) #[K, -1]
        for layer in self.hand_fcs:
            hand_features = F.relu(layer(hand_features))

        K_spatial_attention_scores = []
        for feature in hand_object_features:
            scores = self.spatial_attention(feature) #[M_i, 4]
            K_spatial_attention_scores.append(scores)

        K_hand_object_features = [] 
        for feature in hand_object_features:
            feature = torch.flatten(feature, start_dim=1) #[M, -1] 
            for layer in self.hand_object_fcs:
                feature = F.relu(layer(feature))
            K_hand_object_features.append(feature)

        num_hands = hand_features.shape[0]
        K_scores = []
        for i in range(num_hands):
            if hand_object_features[i].shape[0]:
                hand_ftr = hand_features[i:i+1].repeat(hand_object_features[i].shape[0], 1) #[M_i, -1]
                projected_features = torch.cat(
                    [
                        hand_ftr, K_cross_attention_hands_fcs[i], K_hand_object_features[i]     
                    ], dim=1
                ) #[M_i, -1]
                for layer in self.project_fcs:
                    projected_features = layer(projected_features)
                
                scores = self.classifier(projected_features) #[M_i, 4]
                scores = scores + K_spatial_attention_scores[i]
                scores = torch.max(scores, dim=0).values.unsqueeze(0) # MIL with max operaton, shape #[1, 4]
                K_scores.append(scores)
            else:
                projected_features = torch.cat([hand_features[i:i+1] for j in range(3)], dim=1)
                for layer in self.project_fcs:
                    projected_features = layer(projected_features)
                scores = self.classifier(projected_features)
                K_scores.append(scores)

        if num_hands:
            out = torch.cat(K_scores, dim=0)
        else:
            out = hand_features.view(0, 4)

        if self.training:
            return {"loss_contact": contact_loss(out, instances, self.pos_weight, self.device)}
        else:
            contact_head_inference(out, instances)
            return instances 

def build_contact_head(cfg, input_shape):
    """
    Build a contact head defined by `cfg.MODEL.ROI_CONTACT_HEAD.NAME`
    """
    name = cfg.MODEL.ROI_CONTACT_HEAD.NAME 
    return ROI_CONTACT_HEAD_REGISTRY.get(name)(cfg, input_shape)        