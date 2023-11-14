#!/bin/bash

#SBATCH --job-name=v500-GAN
#SBATCH --output=out_GAN_500.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-500' --n_downsamplings 5 --encoded_size 24 --adv_train True --cGAN True --batch_size 2 --epochs 100 --epoch_decay 100 --lr 0.00025 --ls_reg_weight 2e-6