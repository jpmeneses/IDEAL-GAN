#!/bin/bash

#SBATCH --job-name=v036-sup
#SBATCH --output=out_sup_036.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-036' --adv_train True --n_G_filters 36 --encoded_size 128 --epochs 100 --epoch_decay 100 --epoch_ckpt 20 --cycle_loss_weight 5e0 --ls_reg_weight 1e-5 --Fourier_reg_weight 0.0 --NL_SelfAttention True