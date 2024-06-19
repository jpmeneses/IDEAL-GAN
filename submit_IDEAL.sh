#!/bin/bash

#SBATCH --job-name=v105-unsup
#SBATCH --output=out_unsup_105.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-IDEAL-unsup.py --dataset Unsup-105 --rand_ne False --UQ True --epochs 40 