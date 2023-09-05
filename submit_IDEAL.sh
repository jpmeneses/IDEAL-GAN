#!/bin/bash

#SBATCH --job-name=v110-Unsup
#SBATCH --output=out_unsup_110.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-110' --out_vars 'FM' --UQ True --k_fold 5 --epochs 35 --epoch_decay 35