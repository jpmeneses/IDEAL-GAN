#!/bin/bash

#SBATCH --job-name=v102-GAN
#SBATCH --output=out_GAN_102.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-102' --adv_train False --n_G_filters 36 --n_downsamplings 3 --encoded_size 6 --epochs 100 --epoch_decay 100 --epoch_ckpt 10 --cycle_loss_weight 1e1 --ls_reg_weight 1e-6 --Fourier_reg_weight 0.0 --NL_SelfAttention True