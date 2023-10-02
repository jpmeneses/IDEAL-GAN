#!/bin/bash

#SBATCH --job-name=v302-GAN
#SBATCH --output=out_GAN_302.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-302' --n_downsamplings 3 --n_res_blocks 3 --n_groups_PM 2 --encoded_size 24 --VQ_encoder True --adv_train True --cGAN True --epochs 50 --epoch_decay 50 --epoch_ckpt 10