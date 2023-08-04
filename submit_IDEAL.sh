#!/bin/bash

#SBATCH --job-name=v046-GAN
#SBATCH --output=out_GAN_046.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-046' --adv_train True --n_G_filters 36 --encoded_size 128 --epochs 60 --epoch_decay 60 --epoch_ckpt 5 --D_lr_factor 4 --critic_train_steps 2 --cycle_loss_weight 1e1 --ls_reg_weight 1e-5 --Fourier_reg_weight 0.0 --NL_SelfAttention True