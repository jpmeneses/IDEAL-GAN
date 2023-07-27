#!/bin/bash

#SBATCH --job-name=v044-sup
#SBATCH --output=out_sup_044.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-044' --adv_train True --n_G_filters 36 --encoded_size 128 --epochs 160 --epoch_decay 160 --epoch_ckpt 20 --D_lr_factor 4 --critic_train_steps 3 --cycle_loss_weight 5e0 --ls_reg_weight 5e-6 --Fourier_reg_weight 0.0 --NL_SelfAttention True