# AdaPerFormer: Adaptive Perception Transformer for Temporal Action Localization

## Introduction
This code repo implements AdaPerFormer, described in the technial report: [AdaPerFormer](arxiv)

![AdaPerformer Overview](./AdaPerformer_v2.png)

## Code Overview
The of the main components are：
* ./configs: dataset config.
* ./datasets: Data loader and IO module.
* ./model: Our main model with all its building blocks.
* ./src: Startup script, including train and test.
* ./utils: Utility functions for training, inference and other utils.

## Requirements
- Linux
- Python >= 3.5
- CUDA >= 11.0
- GCC >= 4.9
- Other requirements:
```bash
    pip install -r requirement.txt
```

## Data Preparation
1. Download the original video data from [data](https://www.crcv.ucf.edu/THUMOS14/download.html) and use the [I3D](https://github.com/piergiaj/pytorch-i3d) backbone to extract the features.
2. Place I3D_features into the folder `./data`

* The folder structure should look like
```
This folder
│   README.md
│   ...  
│
└───data/
│    └───thumos14/
│    │	 └───i3d_features   
│    └───...
```
## Training and Evaluation
* Train our ActionFormer with I3D features. This will create a experiment folder under *./ckpt* that stores training config, logs, and checkpoints.

```shell
./src/train_thumos.sh
```
* [Optional] Monitor the training using TensorBoard
```shell
tensorboard --logdir=./ckpt/thumos_i3d_reproduce/logs
```
* Evaluate the trained model. 
```shell
./src/test_thumos.sh
```