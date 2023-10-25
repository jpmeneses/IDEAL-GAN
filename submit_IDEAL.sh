#!/bin/bash

#SBATCH --job-name=v201-unsup
#SBATCH --output=out_unsup_201.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-201' --out_vars 'FM' --UQ True --k_fold 1 --epochs 30 --epoch_decay 30 --epoch_ckpt 10
python train-IDEAL-unsup.py --dataset 'Unsup-201' --out_vars 'R2s' --UQ True --k_fold 1 --epochs 35 --epoch_decay 35 --epoch_ckpt 5
python train-IDEAL-unsup.py --dataset 'Unsup-201' --out_vars 'PM' --UQ True --k_fold 1 --epochs 60 --epoch_decay 60 --epoch_ckpt 20