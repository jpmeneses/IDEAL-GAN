#!/bin/bash

#SBATCH --job-name=v030-sup
#SBATCH --output=out_sup_030.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-030' --adv_train True --n_G_filters 36 --encoded_size 128 --epochs 100 --epoch_ckpt 20 --cycle_loss_weight 2e-1 --NL_SelfAttention True