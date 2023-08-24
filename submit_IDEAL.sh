#!/bin/bash

#SBATCH --job-name=v209-GAN
#SBATCH --output=out_GAN_209.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-209' --adv_train True --n_G_filters 36 --n_downsamplings 4 --n_res_blocks 2 --encoded_size 24 --epochs 100 --epoch_decay 100 --epoch_ckpt 20 --D_lr_factor 4 --data_aug_p 0.0 --critic_train_steps 3 --perceptual_loss False --cycle_loss_weight 1e0 --ls_reg_weight 1e-6 --Fourier_reg_weight 0.0 --NL_SelfAttention True