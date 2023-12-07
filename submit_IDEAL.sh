#!/bin/bash

#SBATCH --job-name=v411-GAN
#SBATCH --output=out_GAN_411.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset GAN-411 --rand_ne True --n_downsamplings 3 --div_decod True --encoded_size 3 --adv_train True --cGAN True --batch_size 8 --epochs 140 --epoch_decay 140 --lr 0.001 --ls_reg_weight 1e-6