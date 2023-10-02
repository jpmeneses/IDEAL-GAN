#!/bin/bash

#SBATCH --job-name=v301-GAN
#SBATCH --output=out_GAN_301.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-301' --n_G_filters 36 --n_downsamplings 4 --n_res_blocks 2 --n_groups_PM 2 --encoded_size 24 --VQ_encoder True --adv_train True --cGAN True --n_D_filters 72 --epochs 50 --epoch_decay 50 --epoch_ckpt 10 --perceptual_loss True --A_loss_weight 1e-2 --B_loss_weight 1e-1