#!/bin/bash

#SBATCH --job-name=v249-GAN
#SBATCH --output=out_GAN_249.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-249' --encoded_size 24 --adv_train True --cGAN True --main_loss 'MAE' --A_loss 'sinGAN' --B_loss_weight 0.01