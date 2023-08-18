#!/bin/bash

#SBATCH --job-name=v057-GAN
#SBATCH --output=out_GAN_057.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-057' --adv_train True --n_G_filters 36 --n_downsamplings 4 --n_res_blocks 2 --encoded_size 128 --epochs 100 --epoch_decay 100 --epoch_ckpt 20 --D_lr_factor 4 --critic_train_steps 2 --perceptual_loss False --cycle_loss_weight 5e0 --ls_reg_weight 9e-6 --Fourier_reg_weight 0.0 --NL_SelfAttention True