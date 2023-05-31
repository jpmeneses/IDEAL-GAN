#!/bin/bash

#SBATCH --job-name=v013-GAN
#SBATCH --output=out_GAN_013.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-013' --n_G_filters 36 --encoded_size 64 --epochs 100 --epoch_ckpt 20 --B2A2B_weight 0.5 --ls_reg_weight 1e-7 --NL_SelfAttention False