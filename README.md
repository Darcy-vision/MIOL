# A fully self-supervised approach for real-time and accurate switchable depth recovery in 3D laparoscopy

This repository contains the source code for our paper.

## :floppy_disk: Data Description

To pretrain the network **SSDNet**, you will need to download the required datasets. 
* [Sceneflow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) (DispNet/FlowNet2.0 dataset subsets are enough)

The structure of the downloaded dataset is as follows

```Shell
├── SceneFlowSubData
    ├── disparity_occlusions
        ├── train
        ├── val
    ├── frames_cleanpass
        ├── train
        ├── val
    ├── frames_disparity
        ├── train
        ├── val
```

To validate the **MIOL** framework, you will need to create your own test data structured as follows
```Shell
├── data
    ├── test_02
        ├── 00000.jpg
        ├── 00001.jpg
        ├── ...
        ├── 00600.jpg
        ├── calib.yaml
    ├── test_03
        ├── 00000.jpg
        ├── 00001.jpg
        ├── ...
        ├── 00600.jpg
```

## :sunny: Setup
We recommend using Anaconda to set up an environment
```Shell
cd MIOL
conda create -n pytorch python=3.7
conda activate pytorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```
Please modify the torch version based on your GPU. We managed to test our code on:
- Ubuntu 18.04/20.04 with Python 3.7 and CUDA 11.1.
- Windows 10/11 with Python 3.7 and CUDA 11.7.

## :bicyclist: Training

Pretrained models can be downloaded from [OneDrive](https://mailhfuteducn-my.sharepoint.com/:u:/g/personal/zjylearn_mail_hfut_edu_cn/Ecpdt7UQHWFOue7sr1QX7PoBQFYSESp2qO80VYDtubUN_g?e=T36W1c).

Our model is trained on one RTX-3090 GPU using the following command. Training logs will be written to `checkpoints/log_name` which can be visualized using tensorboard.


```Shell
python train_STM_sceneflow_meta.py --meta-batch-size 4 -k 1 -q 1 --inner-lr 1e-4 --meta-lr 1e-4 --epochs 25 --data path_to_SceneFlowSubData/ --name log_name
```

## :movie_camera: Demos

You can demo the trained model on a sequence of stereo images. To predict depth for your dataset, run
```Shell
python test_inference.py --pretrained-model sceneflow_pretrained.tar --calib-path path_to_calib_yaml --dataset-dir path_to_test_02 --output-dir output --output-depth
```
To save the depth values as `.npy` files and the depth maps as `.png` images, run with the `--output-depth` flag. 

## :candy: Visualization
Here we show some point cloud results of the proposed **MIOL** framework on [Hamlyn](https://arxiv.org/abs/1705.08260) and [SCARED](https://arxiv.org/abs/2101.01133) datasets.


https://user-images.githubusercontent.com/131570332/233920820-c1057d58-0803-44a3-b8ca-49aa056ab538.mp4


## :rose: Acknowledgment
- Our code is based on [SC-SfMLearner](https://github.com/JiawangBian/SC-SfMLearner-Release).
