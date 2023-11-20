#!/bin/bash

#SBATCH --job-name=v203-TEaug
#SBATCH --output=out_TEaug_203.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-TEaug.py --dataset 'TEaug-203' --data_size 384 --n_echoes 3 --out_vars PM --n_G_filters 36