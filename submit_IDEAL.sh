#!/bin/bash

#SBATCH --job-name=v023-GAN
#SBATCH --output=out_GAN_023.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-023' --adv_train False --n_G_filters 36 --encoded_size 128 --epochs 100 --epoch_ckpt 20 --cycle_loss_weight 5e1 --ls_reg_weight 5e-5 --NL_SelfAttention True