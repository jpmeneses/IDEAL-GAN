#!/bin/bash

#SBATCH --job-name=v101-uns
#SBATCH --output=out_uns_101.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-IDEAL-unsup.py --dataset Unsup-101 --UQ True --epochs 40 