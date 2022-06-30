#!/bin/bash

#SBATCH --job-name=v006-IDEALGAN
#SBATCH --output=out_IDEALGAN_006.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL.py --dataset 'IDEAL-GAN-006' --G_model 'encod-decod' --n_G_filters 32 --batch_size 4 --epochs 60 --epoch_decay 60 --epoch_ckpt 5 --lr 0.0002 --FM_fix True