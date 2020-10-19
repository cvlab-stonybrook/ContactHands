# Installation

This code is based on [Detectron2](https://github.com/facebookresearch/detectron2).

Use the following instructions.

#### Create a new conda environment:

```conda create -n detectron2_v0.1.1 python=3.7```

```conda activate detectron2_v0.1.1```

#### Install the following dependencies:

```conda install pytorch torchvision cudatoolkit=10.0 -c pytorch```

```python -m pip install python-dateutil>=2.1 pycocotools>=2.0.1```

```python -m pip install opencv-python ipython scipy scikit-image```

#### Clone Detectron2 v0.1.1 and install the following commit (using other commit can give errors):

```git clone https://github.com/facebookresearch/detectron2.git --branch v0.1.1 detectron2_v0.1.1```

```cd detectron2_v0.1.1```

```git checkout db1614e```

```python -m pip install -e```

#### Clone this ContactHands repository:

```git clone https://github.com/cvlab-stonybrook/ContactHands.git```

```cd ContactHands```
