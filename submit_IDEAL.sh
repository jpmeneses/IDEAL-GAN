#!/bin/bash

#SBATCH --job-name=v119-unsup
#SBATCH --output=out_unsup_119.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-119' --n_echoes 5 --out_vars 'R2s' --UQ True --k_fold 4 --epochs 30 --epoch_decay 30 --epoch_ckpt 5
python train-IDEAL-unsup.py --dataset 'Unsup-119' --n_echoes 5 --out_vars 'PM' --UQ True --k_fold 4 --epochs 60 --epoch_decay 60 --epoch_ckpt 20