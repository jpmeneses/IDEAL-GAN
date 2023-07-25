#!/bin/bash

#SBATCH --job-name=v041-sup
#SBATCH --output=out_sup_041.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-041' --adv_train True --n_G_filters 36 --encoded_size 128 --epochs 60 --epoch_decay 60 --epoch_ckpt 20 --cycle_loss_weight 4e0 --ls_reg_weight 5e-6 --Fourier_reg_weight 0.0 --NL_SelfAttention True