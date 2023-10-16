#!/bin/bash

#SBATCH --job-name=v308-GAN
#SBATCH --output=out_GAN_308.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-308' --n_downsampling 3 --n_res_blocks 3 --encoded_size 1 --VQ_encoder True --VQ_num_embed 32 --adv_train True --cGAN True --epochs 80 --epoch_decay 80 --epoch_ckpt 20 --ls_reg_weight 4e1