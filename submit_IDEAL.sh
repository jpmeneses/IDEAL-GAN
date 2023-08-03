#!/bin/bash

#SBATCH --job-name=v100-sup
#SBATCH --output=out_sup_100.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-100' --adv_train False --n_G_filters 36 --encoded_size 128 --epochs 50 --epoch_decay 50 --epoch_ckpt 10 --cycle_loss_weight 1e1 --ls_reg_weight 1e-6 --Fourier_reg_weight 0.0 --NL_SelfAttention True