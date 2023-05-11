#!/bin/bash

#SBATCH --job-name=v003-GAN
#SBATCH --output=out_GAN_003.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'TEaug-003' --n_G_filters 48 --epochs 100 --cycle_loss_weight 0.5 --D1_SelfAttention True