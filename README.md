##### Table of contents
1. [Installation](#Installation)
2. [Data Preparation](#Data-Preparation)
3. [Training and Testing](#Training-and-Testing) 
4. [Quick Demo](#Quick-Demo)
5. [Acknowledgments](#Acknowledgments)
6. [Contacts](#Contacts)

# ISBNet: a 3D Point Cloud Instance Segmentation Network with Instance-aware Sampling and Box-aware Dynamic Convolution

<a href="https://arxiv.org/abs/2303.00246"><img src="https://img.shields.io/badge/https%3A%2F%2Farxiv.org%2Fabs%2F2303.00246-arxiv-brightgreen"></a>

[Tuan Duc Ngo](https://ngoductuanlhp.github.io/),
[Binh-Son Hua](https://sonhua.github.io/),
[Khoi Nguyen](https://www.khoinguyen.org/)<br>
VinAI Research, Vietnam

> **Abstract**: 
Existing 3D instance segmentation methods are predominant by a bottom-up design: a manually fine-tuned algorithm to group points into clusters followed by a refinement network. Relying on the quality of the clusters, these methods generate susceptible results when (1) nearby objects with the same semantic class are packed together, or (2) large objects with complex shapes. To address these shortcomings, we introduce ISBNet, a novel cluster-free method that represents instances as kernels and decodes instance masks via dynamic convolution. To efficiently generate a high-recall and discriminative kernel set, we propose a simple strategy, named Instance-aware Farthest Point Sampling, to sample candidates and leverage the point aggregation layer adopted from PointNet++ to encode candidate features. Moreover, we show that training 3D instance segmentation in a multi-task learning setting with an additional axis-aligned bounding box prediction head further boosts performance. Our method set new state-of-the-art results on ScanNetV2 (55.9), S3DIS (60.8), and STPLS3D (49.2) in terms of AP and retains fast inference time (237ms per scene on ScanNetV2).
![overview](docs/isbnet_arch.png)

Details of the model architecture and experimental results can be found in [our paper](https://arxiv.org/abs/2303.00246v1):

```bibtext
@inproceedings{ngo2022geoformer,
 author={Tuan Duc Ngo, Binh-Son Hua, Khoi Nguyen},
 booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
 title={ISBNet: a 3D Point Cloud Instance Segmentation Network with Instance-aware Sampling and Box-aware Dynamic Convolution},
 year= {2023}
}
```

**Please CITE** our paper whenever this repository is used to help produce published results or incorporated into other software.

## Feature
* State of the art performance on ScanNetV2, S3DIS, and STPLS3D.
* High speed of 237ms per scan on ScanNetV2 dataset.
* Reproducibility code for both ScanNetV2, S3DIS and STPLS3D datasets.

## Dataset

- [x] ScanNetV2
- [x] S3DIS
- [ ] STPLS3D
- [ ] ScanNetV2-200

## Installation
Please refer to [installation guide](docs/INSTALL.md).

## Data Preparation
Please refer to [data preparation](docs/DATA_PREPARATION.md).

## Training and Testing
Please refer to [training guide](docs/TRAIN.md).

## Quick Demo

We provide pre-trained models on ScanNetV2 validation set ([GoogleDrive](https://drive.google.com/file/d/1-GQpYlcVRV5r6qDg-Z7_90CIIfu4kmq8/view?usp=sharing)) and S3DIS Area 5 validation ([GoogleDrive](https://drive.google.com/file/d/1oup4nEdgsmdwnMP1TQPmoIqZ8c1RoTgA/view?usp=sharing)).

1\) ScanNetV2 validation set:

```
python3 tools/test.py configs/scannetv2/isbnet_scannetv2.yaml pretrains/scannetv2/best_head.pth
```

2\) S3DIS Area5 validation:

```
python3 tools/test.py configs/s3dis/isbnet_s3dis_area5.yaml pretrains/s3dis/best_head_val_area5.pth
```

## Acknowledgements
This repo is built upon [spconv](https://github.com/traveller59/spconv), [DyCo3D](https://github.com/aim-uofa/DyCo3D), [SoftGroup](https://github.com/thangvubk/SoftGroup). 

## Contacts
If you have any questions or suggestions about this repo, please feel free to contact me (ductuan.ngo99@gmail.com).