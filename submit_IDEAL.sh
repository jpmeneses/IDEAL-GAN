#!/bin/bash

#SBATCH --job-name=v107-unsup
#SBATCH --output=out_unsup_107.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-IDEAL-unsup.py --dataset Unsup-107 --rand_ne False --out_vars R2s --UQ True --epochs 70