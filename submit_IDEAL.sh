#!/bin/bash

#SBATCH --job-name=v008-GAN
#SBATCH --output=out_GAN_008.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-008' --n_G_filters 24 --encoded_size 256 --epochs 100 --cycle_loss_weight 0.5 --ls_reg_weight 10.0 --D1_SelfAttention True