#!/bin/bash

#SBATCH --job-name=v105-TEaug
#SBATCH --output=out_TEaug_105.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-TEaug.py --dataset 'TEaug-105' --field 3.0 --out_vars 'PM' --epochs 140 --epoch_decay 140 --epoch_ckpt 20