#!/bin/bash

#SBATCH --job-name=v244-GAN
#SBATCH --output=out_GAN_244.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-244' --encoded_size 24 --adv_train True --cGAN True