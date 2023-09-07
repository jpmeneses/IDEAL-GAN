#!/bin/bash

#SBATCH --job-name=v113-GAN
#SBATCH --output=out_GAN_113.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-113' --out_vars 'FM' --UQ True --k_fold 3 --epochs 25 --epoch_decay 25 --epoch_ckpt 5