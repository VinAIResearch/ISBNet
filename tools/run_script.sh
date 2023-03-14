#!/bin/bash -e
#SBATCH --job-name=detr_pc
#SBATCH --output=/lustre/scratch/client/vinai/users/tuannd42/fewshot_ws/slurm_out/slurm_%A.out
#SBATCH --error=/lustre/scratch/client/vinai/users/tuannd42/fewshot_ws/slurm_out/slurm_%A.err

#SBATCH --gpus=1
#SBATCH --nodes=1

#SBATCH --mem-per-gpu=40G
#SBATCH --nodelist=sdc2-hpc-dgx-a100-002
#SBATCH --cpus-per-gpu=32

#SBATCH --partition=applied
#SBATCH --mail-type=all
#SBATCH --mail-user=v.tuannd42@vinai.io


srun --container-image=/lustre/scratch/client/vinai/users/tuannd42/docker_images/geo3dis_new4.sqsh \
--container-mounts=/lustre/scratch/client/vinai/users/tuannd42/fewshot_ws/Geo3DIS:/home/ubuntu/Geo3DIS \
--container-workdir=/home/ubuntu/Geo3DIS/ \
python3 tools/train.py configs/s3dis/s3dis_spp_pool_nofilterbg.yaml --trainall --exp_name head_trainall_hardfilter

# srun --partition=applied \
# --job-name=spf \
# --pty \
# --nodes=1 \
# --gpus-per-node=1 \
# --mem-per-gpu=40G \
# --cpus-per-gpu=32 \
# --container-image=/lustre/scratch/client/vinai/users/tuannd42/docker_images/geo3dis_new4.sqsh \
# --container-mounts=/lustre/scratch/client/vinai/users/tuannd42/fewshot_ws/Geo3DIS:/home/ubuntu/Geo3DIS \
# --container-workdir=/home/ubuntu/Geo3DIS/ \
# /bin/bash