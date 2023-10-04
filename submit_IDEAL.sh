#!/bin/bash

#SBATCH --job-name=v245-GAN
#SBATCH --output=out_GAN_245.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-245' --encoded_size 24 --adv_train True --cGAN True --perceptual_loss False --A_loss_weight 0.0 --ls_reg_weight 1e-7 --Fourier_reg_weight 1e-4