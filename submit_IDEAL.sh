#!/bin/bash

#SBATCH --job-name=v002-GAN
#SBATCH --output=out_GAN_002.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-002' --epochs 50 --epoch_decay 50 --epoch_ckpt 5 --cycle_loss_weight 1.0