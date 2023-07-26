#!/bin/bash

#SBATCH --job-name=v043-sup
#SBATCH --output=out_sup_043.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-043' --adv_train True --n_G_filters 36 --encoded_size 128 --epochs 160 --epoch_decay 160 --epoch_ckpt 20 --critic_train_steps 5 --cycle_loss_weight 5e0 --ls_reg_weight 5e-6 --Fourier_reg_weight 0.0 --NL_SelfAttention True