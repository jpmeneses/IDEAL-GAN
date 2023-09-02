#!/bin/bash

#SBATCH --job-name=v100-Unsup
#SBATCH --output=out_unsup_100.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-100' --UQ True --epochs 35 --epoch_decay 35