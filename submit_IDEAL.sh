#!/bin/bash

#SBATCH --job-name=v108-Unsup
#SBATCH --output=out_unsup_108.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-108' --out_vars 'PM' --UQ True --k_fold 3 --epochs 100 --epoch_decay 100