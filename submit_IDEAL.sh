#!/bin/bash

#SBATCH --job-name=v105-Unsup
#SBATCH --output=out_unsup_105.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-105' --out_vars 'PM' --UQ True --k_fold 5 --epochs 35 --epoch_decay 35 --FM_L1_weight 1e-2