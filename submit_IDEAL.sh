#!/bin/bash

#SBATCH --job-name=v112-GAN
#SBATCH --output=out_GAN_112.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-112' --out_vars 'FM' --UQ True --k_fold 2 --epochs 20 --epoch_decay 20