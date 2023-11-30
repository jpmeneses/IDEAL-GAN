#!/bin/bash

#SBATCH --job-name=v603-GAN
#SBATCH --output=out_603_GAN.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset GAN-603 --n_downsamplings 3 --encoded_size 3 --VQ_encoder True --VQ_num_embed 32 --adv_train True --cGAN True --batch_size 8 --epochs 160 --epoch_decay 160 --lr 0.001 --ls_reg_weight 5.0