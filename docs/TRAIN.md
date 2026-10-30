# Training and Testing guide

## Training

1\) ScanNetV2 dataset

Pretrain the 3D Unet backbone from scratch

```
python3 tools/train.py configs/scannetv2/isbnet_backbone_scannetv2.yaml --only_backbone --exp_name pretrain_backbone
```

We also provided the pre-trained 3D backbone models of [ScanNetV2 val](https://huggingface.co/TuanNgo/ISBNet/resolve/main/scannetv2/pretrain_scannetv2_val.pth), [ScanNetV2-200 val](https://huggingface.co/TuanNgo/ISBNet/resolve/main/scannet200/pretrain_scannetv2_200_val.pth), [S3DIS val Area 5](https://huggingface.co/TuanNgo/ISBNet/resolve/main/s3dis/pretrain_s3dis_area5.pth), and [STPLS3D val](https://huggingface.co/TuanNgo/ISBNet/resolve/main/stpls3d/pretrain_stpls3d_val.pth). Alternatively, we can also finetune other pre-trained backbones from [SoftGroup](https://github.com/thangvubk/SoftGroup) or [SSTNet](https://github.com/Gorilla-Lab-SCUT/SSTNet) with fewer epoches by set the pretrain path in the config file to the corresponding pre-trained weights.

Train our ISBNet

```
python3 tools/train.py configs/scannetv2/isbnet_scannetv2.yaml --trainall --exp_name default
```

By default, we set `batch_size=12` on a single V100 GPU.

2\) S3DIS dataset

```
# Pretrain step
python3 tools/train.py configs/s3dis/isbnet_backbone_s3dis_area5.yaml --only_backbone  --exp_name default

# Train entire model
python3 tools/train.py configs/s3dis/isbnet_s3dis_area5.yaml --trainall  --exp_name default
```

3\) STPLS3D dataset

```
# Pretrain step
python3 tools/train.py configs/stpls3d/isbnet_backbone_stpls3d.yaml --only_backbone  --exp_name default

# Train entire model
python3 tools/train.py configs/stpls3d/isbnet_stpls3d.yaml --trainall  --exp_name default
```

## Inference

1\) For evaluation (on ScanNetV2 val, S3DIS, and STPLS3D)

```
python3 tools/test.py configs/<config_file> <checkpoint_file>
```

2\) For exporting predictions (i.e., to submit results to ScanNetV2 hidden benchmark)

```
python3 tools/test.py configs/<config_file> <checkpoint_file> --out <output_dir>
```