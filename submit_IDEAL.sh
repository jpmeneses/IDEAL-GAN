#!/bin/bash

#SBATCH --job-name=v102-uns
#SBATCH --output=out_uns_102.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-IDEAL-unsup.py --dataset Unsup-102 --UQ True --epochs 40 