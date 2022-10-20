#!/bin/bash

#SBATCH --job-name=v011-unsup
#SBATCH --output=out_unsup_011.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-011' --UQ True --epochs 25 --epoch_decay 25 --epoch_ckpt 5 --lr 0.0001 --beta_1 0.9 --beta_2 0.999