#!/bin/bash

#SBATCH --job-name=v104-unsup
#SBATCH --output=out_unsup_104.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-IDEAL-unsup.py --dataset Unsup-104 --rand_ne True --UQ True --epochs 40 