#!/bin/bash

#SBATCH --job-name=v001-GAN
#SBATCH --output=out_GAN_001.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-001' --epochs 50 --epoch_decay 50 --epoch_ckpt 5