#!/bin/bash

#SBATCH --job-name=v400-GAN
#SBATCH --output=out_GAN_400.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

ython train-IDEAL-GAN.py --dataset 'GAN-400' --encoded_size 24 --adv_train True --cGAN True --batch_size 8 --lr 0.001