#!/bin/bash

#SBATCH --job-name=v115-GAN
#SBATCH --output=out_GAN_115.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-115' --out_vars 'R2s' --UQ True --k_fold 5 --epochs 25 --epoch_decay 25 --epoch_ckpt 5