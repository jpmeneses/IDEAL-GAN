#!/bin/bash

#SBATCH --job-name=v109-unsup
#SBATCH --output=out_unsup_109.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-IDEAL-unsup.py --dataset Unsup-109 --rand_ne False --out_vars PM --UQ True --epochs 40 --beta_1 0.5 --beta_2 0.9