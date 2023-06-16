#!/bin/bash

#SBATCH --job-name=v020-GAN
#SBATCH --output=out_GAN_020.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-020' --n_G_filters 36 --encoded_size 128 --epochs 140 --epoch_decay 140 --epoch_ckpt 20 --cycle_loss_weight 1e1 --ls_reg_weight 1e-6 --NL_SelfAttention True