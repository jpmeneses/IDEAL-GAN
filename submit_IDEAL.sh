#!/bin/bash

#SBATCH --job-name=v306-GAN
#SBATCH --output=out_GAN_306.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-306' --n_downsamplings 3 --n_res_blocks 3 --encoded_size 2 --VQ_encoder True --VQ_num_embed 32 --adv_train True --cGAN True --epochs 80 --epoch_decay 80 --epoch_ckpt 20 --ls_reg_weight 1e1