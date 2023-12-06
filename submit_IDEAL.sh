#!/bin/bash

#SBATCH --job-name=v410-GAN
#SBATCH --output=out_410_GAN.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset GAN-410 --rand_ne True --encoded_size 24 --adv_train True --cGAN True --batch_size 8 --lr 0.001 --ls_reg_weight 2e-6