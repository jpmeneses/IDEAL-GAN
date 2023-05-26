#!/bin/bash

#SBATCH --job-name=v001-GAN
#SBATCH --output=out_GAN_001.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-001' --n_G_filters 24 --encoded_size 4096 --epochs 100 --epoch_ckpt 20 --B2A2B_weight 0.5 --ls_reg_weight 0.1 --D1_SelfAttention False