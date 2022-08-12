#!/bin/bash

#SBATCH --job-name=v003-TEaug
#SBATCH --output=out_TEaug_003.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-TEaug.py --dataset 'TEaug-003' --n_echoes 3 --epochs 100 --epoch_decay 100 --lr 0.0001 --beta_1 0.9 --beta_2 0.999