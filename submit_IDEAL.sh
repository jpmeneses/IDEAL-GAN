#!/bin/bash

#SBATCH --job-name=v117-unsup
#SBATCH --output=out_unsup_117.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-117' --n_echoes 5 --out_vars 'FM' --UQ True --k_fold 2 --epochs 20 --epoch_decay 20 --epoch_ckpt 10