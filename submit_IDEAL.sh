#!/bin/bash

#SBATCH --job-name=v407-GAN
#SBATCH --output=out_GAN_407.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-407' --encoded_size 24 --adv_train True --cGAN True --batch_size 8 --epochs 200 --epoch_decay 200 --lr 0.001 --ls_reg_weight 2e-6