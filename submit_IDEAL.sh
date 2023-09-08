#!/bin/bash

#SBATCH --job-name=v118-unsup
#SBATCH --output=out_unsup_118.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-118' --n_echoes 5 --out_vars 'PM' --UQ True --k_fold 3 --epochs 60 --epoch_decay 60 --epoch_ckpt 20