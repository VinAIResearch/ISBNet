# Training and Testing guide

## Training

1\) Training ScanNetV2 dataset

Pretrain the 3D Unet backbone

```
python3 tools/train.py configs/scannetv2/isbnet_scannetv2.yaml --only_backbone --exp_name pretrain_backbone
```

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