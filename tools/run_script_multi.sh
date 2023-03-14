#!/bin/bash -e
#SBATCH --job-name=detr_pc
#SBATCH --output=/lustre/scratch/client/vinai/users/tuannd42/fewshot_ws/slurm_out/slurm_%A.out
#SBATCH --error=/lustre/scratch/client/vinai/users/tuannd42/fewshot_ws/slurm_out/slurm_%A.err

#SBATCH --gpus=2
#SBATCH --nodes=1

#SBATCH --mem-per-gpu=40G
#SBATCH --nodelist=sdc2-hpc-dgx-a100-009
#SBATCH --cpus-per-gpu=32

#SBATCH --partition=applied
#SBATCH --mail-type=all
#SBATCH --mail-user=v.tuannd42@vinai.io


srun --container-image=/lustre/scratch/client/vinai/users/tuannd42/docker_images/geo3dis_new4.sqsh \
--container-mounts=/lustre/scratch/client/vinai/users/tuannd42/fewshot_ws/Geo3DIS:/home/ubuntu/Geo3DIS \
--container-workdir=/home/ubuntu/Geo3DIS/ \
python3 -m torch.distributed.run --nproc_per_node=2 --master_port=$((RANDOM + 10000)) tools/train.py --dist configs/scannetv2/lightweight.yaml --only_backbone --exp_name pretrain_lighweight