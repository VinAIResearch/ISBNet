# Training and Testing guide

## Training

1\) Training ScanNetV2 dataset

Pretrain the 3D Unet backbone from scratch

```
python3 tools/train.py configs/scannetv2/isbnet_backbone_scannetv2.yaml --only_backbone --exp_name pretrain_backbone
```

We also provided the pre-trained 3D backbone models of ScanNetV2 ([GoogleDrive](https://drive.google.com/file/d/1DQiMOsZpr9PgaKx9aK8rJhhIvh0vodWd/view?usp=sharing)) and S3DIS ([GoogleDrive](https://drive.google.com/file/d/1SHqrtrb94HQMa4Ml6X6_4JHZbRDqlTEv/view?usp=sharing)). This model achieves ~70.8 mIoU on ScanNet validation set and ~69.0 mIoU on S3DIS Area5 validation. Additionally, we can also finetune other backbones from [SoftGroup](https://github.com/thangvubk/SoftGroup) or [SSTNet](https://github.com/Gorilla-Lab-SCUT/SSTNet) with fewer epoches by set the pretrain path in the config file to the corresponding pre-trained weights.

Train our ISBNet:

```
python3 tools/train.py configs/scannetv2/isbnet_scannetv2.yaml --trainall --exp_name default
```

2\) Training S3DIS dataset

```
python3 tools/train.py configs/s3dis/isbnet_s3dis_area5.yaml --trainall  --exp_name default
```

3\) Training STPLS3D dataset

```
python3 tools/train.py configs/s3dis/isbnet_stpls3d.yaml --trainall  --exp_name default
```

## Inference

1\) For evaluation (on ScanNetV2 val, S3DIS, and STPLS3D)

```
python3 tools/test.py configs/<config_file> <checkpoint_file>
```

2\) For export results (i.e., ScanNetV2 test)

```
python3 tools/test.py configs/<config_file> <checkpoint_file> --out <output_dir>
```