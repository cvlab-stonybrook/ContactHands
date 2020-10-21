# ContactHands

This repository contains the code and data for the following paper:

[Detecting Hands and Recognizing Physical Contact in the Wild](https://www3.cs.stonybrook.edu/~sunarasimhas/webpage/contacthands.pdf) (NeurIPS 2020).

## Contents

This repository contains the following:

* A PyTorch implementation of the proposed architecture for joint hand detection and contact recognition.
* Code to train and evaluate the proposed method.
* The proposed ContactHands dataset.
* Pre-trained models.

### Installation

Follow the instllation instructions in [INSTALL.md](INSTALL.md). 

### Folder structure

The code is organized in the following structure:

```
ContactHands/  
  configs/
    ContactHands.yaml
    Base-RCNN-FPN.yaml
    mask_rcnn_R_101_FPN_3x.yaml
    second_stream.yaml
    
  contact_hands_two_stream/
    config/
      config.py
    data/
      build.py
      dataset_mapper.py
      detection_utils.py
    engine/
      custom_arg_parser.py
      custom_predictor.py
    evaluation/
      evaluator_ourdata.py
    models/
      model_contacthands.pth
      model_best.pth
    modeling/
      metaarch/
        first_stream_rcnn.py
        second_stream_rcnn.py
      roi_heads/
        contact_head.py
        cross_feature_affinity_pooling.py
        first_stream_roi_head.py
        second_stream_roi_head.py
        spatial_attention.py
    utils/
      extend_util_boxes.py
      visualzier.py
  
  datasets/
    load_data.py
  
  output_visualizations/
  
  train_net.py
  detect.py
```

### Data format

Please see the ContactHands dataset (will be out soon!) for the required data and annotation format.

### Models

Download [models](https://drive.google.com/drive/folders/1YpH6AXdurOb0NDgzcaHmWbLDlsZeXlwr) and place them in ```./models/```.

### Training

Use the following command for training:

`python train_net.py --first-config-file configs/ContactHands.yaml`

### Evaluation

Use the following command for evaluation:

`python train_net.py --first-config-file configs/ContactHands.yaml --eval-only MODEL.WEIGHTS <path to model weights> MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.7`

The parameter ```MODEL.ROI_HEADS.SCORE_THRESH_TEST``` is the threshold for hand detections and can take values in the range [0.0, 1.0]. While a lower threshold can increase the AP, it reduces the precision and gives poor hand detections.

### Running and visualizing results on custom images 

Use the following command to run joint hand detection and contact recognition on custom images:

`python detect.py --image_dir <path to a directory containing images> --ROI_SCORE_THRESH 0.7 --sc 0.5 --pc 0.4 --oc 0.6`

The parameters `sc`, `pc`, `oc` denote thresholds for Self-Contact, Other-Person-Contact, and Object-Contact, respectively. 

The thresholds are in the range [0.0, 0.1]. Lower thresholds increases the recall and higher thresholds increases precision. Choose them according to your need. 

If the predicted contact state score for all three of them are less than the corresponding thresholds, the contact state No-Contact will be choosen. 

The output images with hand detections and contact state visualizations will be stored in `./results/`. 

## References
If you find our code or dataset useful, please cite our work using the following:

```
@inproceedings{contacthands_2020,
  title={Detecting Hands and Recognizing Physical Contact in the Wild},
  author={Supreeth Narasimhaswamy and Trung Nguyen and Minh Hoai},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020},
}
```
