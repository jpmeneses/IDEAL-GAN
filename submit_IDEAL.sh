#!/bin/bash

#SBATCH --job-name=v303-GAN
#SBATCH --output=out_GAN_303.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-303' --n_downsamplings 3 --n_res_blocks 3 --n_groups_PM 2 --encoded_size 6 --VQ_encoder True --adv_train True --cGAN True --epochs 50 --epoch_decay 50 --epoch_ckpt 10