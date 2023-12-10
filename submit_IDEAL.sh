#!/bin/bash

#SBATCH --job-name=v412-GAN
#SBATCH --output=out_GAN_412.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset GAN-412 --rand_ne True --n_downsamplings 3 --div_decod True --encoded_size 3 --ls_mean_activ None --adv_train True --cGAN True --batch_size 8 --epochs 140 --epoch_decay 140 --lr 0.001 --ls_reg_weight 4e-6