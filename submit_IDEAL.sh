#!/bin/bash

#SBATCH --job-name=v008-unsup
#SBATCH --output=out_unsup_008.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-008' --epochs 30 --epoch_decay 30 --epoch_ckpt 5 --lr 0.0001 --beta_1 0.9 --beta_2 0.999