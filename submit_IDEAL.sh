#!/bin/bash

#SBATCH --job-name=v701-GAN
#SBATCH --output=out_GAN_701.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-IDEAL-GAN.py --dataset GAN-701 --rand_ne True --only_mag True --div_decod True  --encoded_size 24 --ls_mean_activ None --adv_train True --cGAN True  --batch_size 8 --epochs 140 --epoch_decay 140 --lr 0.001 --ls_reg_weight 2e-6
