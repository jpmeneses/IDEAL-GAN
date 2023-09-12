#!/bin/bash

#SBATCH --job-name=v230-GAN
#SBATCH --output=out_GAN_230.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-230' --adv_train True --cGAN True --n_G_filters 36 --n_downsamplings 4 --n_res_blocks 2 --encoded_size 24 --epochs 100 --epoch_decay 100 --epoch_ckpt 20 --data_aug_p 0.0 --perceptual_loss True --cycle_loss_weight 1e-1 --B2A2B_weight 1e0 --ls_reg_weight 4e-7 --Fourier_reg_weight 0.0 --NL_SelfAttention True