from detectron2.utils.visualizer import Visualizer, GenericMask, ColorMode
import numpy as np 
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks, Boxes, BoxMode, Keypoints, PolygonMasks, RotatedBoxes

def _create_custom_text_labels(classes, scores, class_names, scores_thresh):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
    Returns:
        list[str] or None
    """
    color_assignment = {}
    color_assignment[0] = 'c'
    color_assignment[1] = 'm'
    color_assignment[2] = 'r'
    color_assignment[3] = 'y'

    labels = None
    if classes is not None and class_names is not None and len(class_names) > 0:
        labels = [class_names[i] for i in classes]
    
    if scores is not None:
        if labels is None:
            labels = [
                "NC:{:.2f}, SC:{:.2f}, PC:{:.2f}, OC:{:.2f}".format(s[0]*100, s[1]*100, s[2]*100, s[3]*100) 
                for s in scores
                ]
        else:
            labels = [
                "NC:{:.2f}, SC:{:.2f}, PC:{:.2f}, OC:{:.2f}".format(s[0]*100, s[1]*100, s[2]*100, s[3]*100) 
                for s in scores
                ]

        id_to_names = {}
        id_to_names[0] = "NC"
        id_to_names[1] = "SC"
        id_to_names[2] = "PC"
        id_to_names[3] = "OC"

        colors = [] 
        labels = []
        for s in scores:
            if (s[1]<scores_thresh[1]) and (s[2]<scores_thresh[2]) and (s[3]<scores_thresh[3]):
                colors.append(color_assignment[0])
                labels.append("")
            else:
                col = []
                scr = []
                lbl = []
                for j in range(1, 4):
                    if s[j] >= scores_thresh[j]:
                        col.append(color_assignment[j])
                        scr.append(s[j])
                        lbl.append(id_to_names[j])
                
                if len(col) == 1:
                    colors.append(col[0])
                    labels.append("")
                else:
                    scr = [-k for k in scr]
                    sorted_indices = [i[0] for i in sorted(enumerate(scr), key=lambda x:x[1])]
                    colors.append(col[sorted_indices[0]])
                    lbl_str = ""
                    for sorted_idx in sorted_indices[1:]:
                        lbl_str += lbl[sorted_idx] + ", "
                    lbl_str = lbl_str[:-2]
                    labels.append(lbl_str)

    return labels, colors

class CustomVisualizer(Visualizer):

    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE, scores_thresh=[0.5, 0.5, 0.5, 0.5]):
        super().__init__(img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE)
        self.scores_thresh = scores_thresh


    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.
        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        contacts = predictions.pred_cats if predictions.has("pred_cats") else None 
        scores = contacts * scores.reshape(-1, 1)        
        
        labels, dis_colors = _create_custom_text_labels(classes, scores, "hand", self.scores_thresh)
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.img = self._create_grayscale_image(
                (predictions.pred_masks.any(dim=0) > 0).numpy()
                if predictions.has("pred_masks")
                else None
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=None,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=dis_colors,
            alpha=alpha,
        )
        return self.output    
    
    def draw_dataset_dict(self, dic):
        """
        Draw annotations/segmentaions in Detectron2 Dataset format.
        Args:
            dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.
        Returns:
            output (VisImage): image object with visualizations.
        """
        annos = dic.get("annotations", None)
        if annos:
            if "segmentation" in annos[0]:
                masks = [x["segmentation"] for x in annos]
            else:
                masks = None
            if "keypoints" in annos[0]:
                keypts = [x["keypoints"] for x in annos]
                keypts = np.array(keypts).reshape(len(annos), -1, 3)
            else:
                keypts = None

            boxes = [BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS) for x in annos]

            labels = [x["cats"] for x in annos]
            colors = None
            if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
                colors = [
                    self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in labels
                ]
            names = self.metadata.get("thing_classes", None)
            if names:
                labels = [str(int(i[0]))+", "+str(int(i[1]))+", "+str(int(i[2]))+", "+str(int(i[3])) for i in labels]
            labels = [
                "{}".format(i) + ("|crowd" if a.get("iscrowd", 0) else "")
                for i, a in zip(labels, annos)
            ]
            self.overlay_instances(
                labels=labels, boxes=boxes, masks=masks, keypoints=keypts, assigned_colors=colors
            )

        sem_seg = dic.get("sem_seg", None)
        if sem_seg is None and "sem_seg_file_name" in dic:
            with PathManager.open(dic["sem_seg_file_name"], "rb") as f:
                sem_seg = Image.open(f)
                sem_seg = np.asarray(sem_seg, dtype="uint8")
        if sem_seg is not None:
            self.draw_sem_seg(sem_seg, area_threshold=0, alpha=0.5)

        pan_seg = dic.get("pan_seg", None)
        if pan_seg is None and "pan_seg_file_name" in dic:
            assert "segments_info" in dic
            with PathManager.open(dic["pan_seg_file_name"], "rb") as f:
                pan_seg = Image.open(f)
                pan_seg = np.asarray(pan_seg)
                from panopticapi.utils import rgb2id

                pan_seg = rgb2id(pan_seg)
            segments_info = dic["segments_info"]
        if pan_seg is not None:
            pan_seg = torch.Tensor(pan_seg)
            self.draw_panoptic_seg_predictions(pan_seg, segments_info, area_threshold=0, alpha=0.5)
        return self.output

