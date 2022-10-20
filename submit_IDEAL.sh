#!/bin/bash

#SBATCH --job-name=v009-unsup
#SBATCH --output=out_unsup_009.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-009' --UQ True --epochs 50 --epoch_decay 50 --epoch_ckpt 10 --lr 0.0001 --beta_1 0.9 --beta_2 0.999