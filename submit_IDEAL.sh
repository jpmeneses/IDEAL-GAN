#!/bin/bash

#SBATCH --job-name=v246-GAN
#SBATCH --output=out_GAN_246.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-246' --encoded_size 24 --adv_train True --cGAN True --data_aug_p 0.4 --perceptual_loss True