#!/bin/bash

#SBATCH --job-name=v103-Unsup
#SBATCH --output=out_unsup_103.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-103' --out_vars 'PM' --UQ True --k_fold 4 --epochs 35 --epoch_decay 35