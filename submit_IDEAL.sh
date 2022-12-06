#!/bin/bash

#SBATCH --job-name=v006-Unsup
#SBATCH --output=out_unsup_006.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-TEaug.py --dataset 'Unsup-006' --R2s True --UQ True