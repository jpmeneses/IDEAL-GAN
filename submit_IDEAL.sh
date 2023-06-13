#!/bin/bash

#SBATCH --job-name=v018-GAN
#SBATCH --output=out_GAN_018.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-018' --n_G_filters 36 --encoded_size 128 --epochs 100 --epoch_ckpt 20 --B2A2B_weight 0.5 --ls_reg_weight 1e-7 --NL_SelfAttention True