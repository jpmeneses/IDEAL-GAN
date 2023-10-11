#!/bin/bash

#SBATCH --job-name=v305-GAN
#SBATCH --output=out_GAN_305.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-305' --n_downsamplings 3 --n_res_blocks 3 --encoded_size 6 --VQ_encoder True --VQ_num_embed 32 --adv_train True --cGAN True --epochs 50 --epoch_decay 50 --epoch_ckpt 10 --ls_reg_weight 1e1