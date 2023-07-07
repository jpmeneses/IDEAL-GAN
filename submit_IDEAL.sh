#!/bin/bash

#SBATCH --job-name=v026-GAN
#SBATCH --output=out_GAN_026.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-026' --adv_train True --n_G_filters 36 --encoded_size 128 --epochs 50 --epoch_ckpt 10 --cycle_loss_weight 1e2 --NL_SelfAttention True