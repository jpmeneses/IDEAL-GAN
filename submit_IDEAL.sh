#!/bin/bash

#SBATCH --job-name=v100-uns
#SBATCH --output=out_uns_100.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-IDEAL-unsup.py --dataset Unsup-100 --UQ True --epochs 40 