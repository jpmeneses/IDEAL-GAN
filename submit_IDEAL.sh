#!/bin/bash

#SBATCH --job-name=v259-GAN
#SBATCH --output=out_GAN_259.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --experiment_dir 'GAN-259' --n_downsamplings 3 --encoded_size 6 --adv_train True --cGAN True --ls_reg_weight 0.9e-7