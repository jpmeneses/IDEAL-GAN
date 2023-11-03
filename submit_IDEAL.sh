#!/bin/bash

#SBATCH --job-name=v408-GAN
#SBATCH --output=out_GAN_408.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-408' --n_downsamplings 3 --encoded_size 3 --adv_train True --cGAN True --batch_size 8 --epochs 100 --epoch_decay 100 --lr 0.001 --ls_reg_weight 1e-6