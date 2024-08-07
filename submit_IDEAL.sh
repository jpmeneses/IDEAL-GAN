#!/bin/bash

#SBATCH --job-name=v110-unsup
#SBATCH --output=out_unsup_110.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-IDEAL-unsup.py --dataset Unsup-110 --rand_ne False --out_vars R2s --UQ True --UQ_R2s True --epochs 45 --beta_1 0.5 --beta_2 0.9