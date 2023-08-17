#!/bin/bash

#SBATCH --job-name=v055-GAN
#SBATCH --output=out_GAN_055.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-055' --adv_train False --n_G_filters 36 --n_downsamplings 4 --n_res_blocks 2 --encoded_size 6 --epochs 100 --epoch_decay 100 --epoch_ckpt 20 --perceptual_loss True --cycle_loss_weight 1e0 --B2A2B_weight 2e-1 --ls_reg_weight 2e-6 --Fourier_reg_weight 0.0 --NL_SelfAttention True