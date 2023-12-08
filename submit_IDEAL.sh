#!/bin/bash

#SBATCH --job-name=v505-GAN
#SBATCH --output=out_GAN_505.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset GAN-505 --data_size 384 --rand_ne True --div_decod True --encoded_size 24 --ls_mean_activ None --adv_train True --cGAN True --batch_size 2 --epochs 140 --epoch_decay 140 --lr 0.00025 --ls_reg_weight 2e-6