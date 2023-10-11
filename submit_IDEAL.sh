#!/bin/bash

#SBATCH --job-name=v252-GAN
#SBATCH --output=out_GAN_252.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-252' --encoded_size 24 --adv_train True --cGAN True --A_loss 'sinGAN' --B_loss_weight 0.02 --FM_loss_weight 2.0