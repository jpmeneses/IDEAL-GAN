#!/bin/bash

#SBATCH --job-name=v009-GAN
#SBATCH --output=out_GAN_009.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-009' --n_G_filters 24 --encoded_size 128 --epochs 100 --epoch_ckpt 20 --B2A2B_weight 0.5 --ls_reg_weight 1e-7 --NL_SelfAttention True