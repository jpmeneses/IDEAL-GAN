#!/bin/bash

#SBATCH --job-name=v048-GAN
#SBATCH --output=out_GAN_048.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-048' --adv_train True --n_G_filters 36 --n_downsamplings 3 --n_res_blocks 2 --encoded_size 6 --epochs 60 --epoch_decay 60 --epoch_ckpt 5 --D_lr_factor 4 --critic_train_steps 2 --perceptual_loss False --cycle_loss_weight 5e0 --ls_reg_weight 5e-6 --Fourier_reg_weight 0.0 --NL_SelfAttention True