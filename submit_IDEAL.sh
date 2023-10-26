#!/bin/bash

#SBATCH --job-name=v261-GAN
#SBATCH --output=out_GAN_261.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-261' --n_downsamplings 3 --encoded_size 6 --adv_train True --cGAN True --batch_size 8 --lr 0.0005 --ls_reg_weight 2e-8