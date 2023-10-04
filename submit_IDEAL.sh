#!/bin/bash

#SBATCH --job-name=v246-GAN
#SBATCH --output=out_GAN_246.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-246' --encoded_size 24 --adv_train True --cGAN True --perceptual_loss True --A_loss_weight 3e-3 --B_loss_weight 0.3