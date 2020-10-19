import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.structures import Boxes
from detectron2.modeling.roi_heads import select_foreground_proposals
from .contact_head import build_contact_head
from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from detectron2.modeling.roi_heads.keypoint_head import build_keypoint_head
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from contact_hands_two_stream.utils.extend_utils_boxes import get_union_box, get_nonzeroiou_unionboxes

@ROI_HEADS_REGISTRY.register()
class FirstStreamROIHeads(StandardROIHeads):
    """
    A Standard ROIHeads which contains an addition of a contact head.
    """
    def __init__(self, cfg, input_shape):
        super(FirstStreamROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg, input_shape)
        self._init_mask_head(cfg, input_shape)
        self._init_keypoint_head(cfg, input_shape)
        self._init_contact_head(cfg, input_shape)

    def _init_box_head(self, cfg, input_shape):
        # fmt: off
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = FastRCNNOutputLayers(
            self.box_head.output_shape, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def _init_mask_head(self, cfg, input_shape):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _init_contact_head(self, cfg, input_shape):
        self.contact_head_on = cfg.MODEL.CONTACT_HEAD_ON 
        if not self.contact_head_on:
            return
        pooler_resolution = cfg.MODEL.ROI_CONTACT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_CONTACT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_CONTACT_HEAD.POOLER_TYPE
        self.contact_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.use_intersection = cfg.MODEL.ROI_CONTACT_HEAD.USE_INTERSECTION 
        in_channels = [input_shape[f].channels for f in self.in_features][0]
        self.contact_head = build_contact_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution))

    def forward(
        self,
        images: ImageList,
        second_stream_outputs: Dict[str, torch.Tensor],
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            losses.update(self._forward_contact(second_stream_outputs, features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(second_stream_outputs, features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, second_stream_outputs, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        instances = self._forward_mask(features, instances)
        instances = self._forward_contact(second_stream_outputs, features, instances)
        return instances

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = outputs.predict_boxes_for_gt_classes()
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances

    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            return self.mask_head(mask_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            return self.mask_head(mask_features, instances)

    def _forward_contact(self, second_stream_outputs, features, instances):
        """
        Forward logic of the contact_head branch.

        Args:
            features (list[Tensor]): #level input features for contact prediction
            instances (list[Instances]): the per-image instances to train/predict contact.
                In training, they can be proposals.
                In inference, they can be predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new field "contact" and return it.
        """
        if not self.contact_head_on:
            return {} if self.training else instances 
        
        features = [features[f] for f in self.in_features]
        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            hand_features, object_features, hand_object_features = self.get_pairwise_features(
                features, proposal_boxes, second_stream_outputs, use_intersection=self.use_intersection
            )
            return self.contact_head(hand_features, hand_object_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            hand_features, object_features, hand_object_features = self.get_pairwise_features(
                features, pred_boxes, second_stream_outputs, use_intersection=self.use_intersection
            )
            return self.contact_head(hand_features, hand_object_features, instances)

    def get_pairwise_features(self, features, hand_boxes, object_boxes, use_intersection=True):
        """
        Assume batch size of 1.
        """
        hand_features = self.contact_pooler(features, hand_boxes) # [B, 7, 7, 256]
        hand_boxes = hand_boxes[0].tensor # There is only one element in the list since batch size=1
        object_boxes = object_boxes[0] # There is only one element in the list since batch size=1

        K_object_features = []
        K_unionbox_features = []
        num_hand_box = hand_boxes.shape[0] # B
        for i in range(num_hand_box):
            hand_box = Boxes(hand_boxes[i:i+1])
            union_boxes, object_boxes = get_union_box(hand_box, object_boxes)
            object_features = self.contact_pooler(features, [object_boxes]) #[K, 7, 7, 256]
            hand_object_features = self.contact_pooler(features, [union_boxes]) #[K, 7, 7, 256]
            K_object_features.append(object_features)
            K_unionbox_features.append(hand_object_features)

        return hand_features, K_object_features, K_unionbox_features