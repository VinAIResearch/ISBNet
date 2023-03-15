#!/bin/bash -e
#SBATCH --job-name=rep
#SBATCH --output=/home/tuannd42/my_ws/3dis_ws/slurm_out/slurm_%A.out
#SBATCH --error=/home/tuannd42/my_ws/3dis_ws/slurm_out/slurm_%A.err

#SBATCH --gpus=1
#SBATCH --nodes=1

#SBATCH --mem-per-gpu=40G
#SBATCH --nodelist=sdc2-hpc-dgx-a100-009
#SBATCH --cpus-per-gpu=32

#SBATCH --partition=applied
#SBATCH --mail-type=all
#SBATCH --mail-user=v.tuannd42@vinai.io


srun --container-image=/home/tuannd42/my_ws/docker_images/isbnet.sqsh \
--container-mounts=/home/tuannd42/my_ws/3dis_ws/ISBNet:/home/ubuntu/ISBNet \
--container-workdir=/home/ubuntu/ISBNet/ \
python3 tools/train.py configs/s3dis/isbnet_s3dis_area5.yaml --trainall --exp_name head_trainall
# python3 tools/train.py configs/scannetv2/isbnet_scannetv2.yaml --trainall --exp_name head_trainall_fp16
# python3 tools/train.py configs/stpls3d/isbnet_backbone_stpls3d.yaml --only_backbone --exp_name pretrain_nocoords_m16
# python3 tools/train.py configs/scannetv2/isbnet_scannetv2.yaml --trainall --exp_name head_trainall