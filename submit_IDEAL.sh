#!/bin/bash

#SBATCH --job-name=v101-Unsup
#SBATCH --output=out_unsup_101.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-101' --out_vars 'PM' --UQ True --k_fold 2 --epochs 35 --epoch_decay 35