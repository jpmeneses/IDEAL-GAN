#!/bin/bash

#SBATCH --job-name=v300-TEa
#SBATCH --output=out_TEa_300.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-IDEAL-TEaug.py --dataset TEaug-300 --data_size 384 --n_echoes 0 --out_vars PM --n_G_filters 36 --batch_size 16 --lr 0.001