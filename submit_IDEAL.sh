#!/bin/bash

#SBATCH --job-name=v119-unsup
#SBATCH --output=out_unsup_119.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-119' --n_echoes 5 --out_vars 'R2s' --UQ True --k_fold 4 --epochs 25 --epoch_decay 25 --epoch_ckpt 5