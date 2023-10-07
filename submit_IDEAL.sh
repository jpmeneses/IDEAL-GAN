#!/bin/bash

#SBATCH --job-name=v247-GAN
#SBATCH --output=out_GAN_247.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-247' --encoded_size 24 --adv_train True --cGAN True --A_loss 'sinGAN'