#!/bin/bash

#SBATCH --job-name=v254-GAN
#SBATCH --output=out_GAN_254.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-254' --encoded_size 24 --adv_train True --cGAN True --A_loss 'sinGAN' --A_loss_weight 0.01 --B_loss_weight 0.005 --FM_loss_weight 2.0