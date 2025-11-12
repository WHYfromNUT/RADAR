# High-Resolution Underwater Creature Segmentation 


> **Authors:** 
> Huiyang Wu,
> Qiuping Jiang,
> Zongwei Wu,
> Runmin Cong,
> Cedric Demonceaux,
> Yi Yang
> and Xiangyang Ji.

## 1. Preface

- This repository provides code for "_**High-Resolution Underwater Creature Segmentation**_" TIP-2025. [![Arxiv Page](https://img.shields.io/badge/Arxiv-2207.00794-red?style=flat-square)](https://arxiv.org/abs/2207.00794)
- Created by Huiyang Wu, email: 2311100185@nbu.edu.cn
## 2. High-Resolution Underwater Creature Segmentation Dataset UCS4K
Baidu Netdisk: [UCS4K](https://pan.baidu.com/s/10p8Z_4-oK38Q76yjwpnalw?pwd=1390) **fetch code**: [1390]  &&&
Google drive: [UCS4K](https://drive.google.com/file/d/1PH0PwKchXnkWwtAwbhNSW4utMCp5zer8/view?usp=sharing) is the first large-scale dataset for High-Resolution Underwater Creature Segmentation (UCS). 
It is free for academic research, not for any commercial purposes.

## 3.Directory
The directory should be like this:

````
-- model (saved model)
-- pre (pretrained model)
-- result (saliency maps)
-- data (train dataset and test dataset)
   |-- trian
   |   |-- Imgs
   |   |-- GT
   |   |-- Edge_gt
   |-- val
   |   |-- Imgs
   |   |-- GT
   |   |-- Edge_gt
   |-- test
   |   |-- Imgs
   |   |-- GT
   |   |-- Edge_gt
   ...
   
````
## 4. Proposed Baseline

### 4.1. Training/Testing

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with 
a single NVIDIA TIAN GPU of 24 GB Memory.

1. Configuring your environment (Prerequisites):
    
    + Creating a virtual environment in terminal: `conda create -n RADAR python=3.8`.
    
    + Installing necessary packages: `pip install -r requirements.txt`.

1. Downloading necessary data:

    + downloading testing dataset and move it into `./data/train/`
    
    + downloading training dataset and move it into `./data/train/`,
    
    + downloading [pretrained weights ](https://download.pytorch.org/models/resnet18-5c106cde.pth) and move it into `./checkpoints/best/RADAR.pth`, 
   
    + downloading [ResNet-18](https://download.pytorch.org/models/resnet18-5c106cde.pth) and [Swin-B-224](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth) as backbone networks, which are saved in pre folder.
   
1. Training Configuration:

    + Assigning your costumed path, like `--train_save` and `--train_path` in `etrain.py`.

1. Testing Configuration:

    + After you download all the pre-trained model and testing dataset, just run `etest.py` to generate the final prediction map: 
    replace your trained model directory (`--pth_path`).

### 2.2 Evaluating your trained model:

One-key evaluation is written in MATLAB code (revised from [link](https://github.com/DengPingFan/CODToolbox)), 
please follow this the instructions in `./eval/main.m` and just run it to generate the evaluation results in.

If you want to speed up the evaluation on GPU, you just need to use the [efficient tool](https://github.com/lartpang/PySODMetrics) by `pip install pysodmetrics`.

Assigning your costumed path, like `method`, `mask_root` and `pred_root` in `eval.py`.

Just run `eval.py` to evaluate the trained model.

> pre-computed maps of BGNet can be found in [download link (Google Drive)](https://drive.google.com/file/d/1vhrAGJI81YAK9YSYgPJer0kxzNEfnRT2/view?usp=share_link).

> pre-computed maps of other comparison methods can be found in [download link (Baidu Pan)](https://pan.baidu.com/s/1dLMqa4tix1gdBN1uWrCPbQ) with Code: yxy9.

## 3. Citation

Please cite our paper if you find the work useful: 

	@inproceedings{sun2022bgnet,
	title={Boundary-Guided Camouflaged Object Detection},
	author={Sun, Yujia and Wang, Shuo and Chen, Chenglizhao and Xiang, Tian-Zhu},
	booktitle={IJCAI},
	pages = "1335--1341",
	year={2022}
	}
