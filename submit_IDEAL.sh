#!/bin/bash

#SBATCH --job-name=v015-Unsup
#SBATCH --output=out_unsup_015.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-015' --epochs 20