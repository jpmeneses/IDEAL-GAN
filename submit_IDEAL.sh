#!/bin/bash

#SBATCH --job-name=v123-unsup
#SBATCH --output=out_unsup_123.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-123' --n_echoes 4 --out_vars 'FM' --UQ True --k_fold 3 --epochs 20 --epoch_decay 20 --epoch_ckpt 10
python train-IDEAL-unsup.py --dataset 'Unsup-123' --n_echoes 4 --out_vars 'R2s' --UQ True --k_fold 3 --epochs 25 --epoch_decay 25 --epoch_ckpt 5
python train-IDEAL-unsup.py --dataset 'Unsup-123' --n_echoes 4 --out_vars 'PM' --UQ True --k_fold 3 --epochs 60 --epoch_decay 60 --epoch_ckpt 20