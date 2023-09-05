#!/bin/bash

#SBATCH --job-name=v106-Unsup
#SBATCH --output=out_unsup_106.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-106' --out_vars 'PM' --UQ True --k_fold 1 --epochs 100 --epoch_decay 100