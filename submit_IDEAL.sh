#!/bin/bash

#SBATCH --job-name=v204-TEaug
#SBATCH --output=out_204_TEaug.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-TEaug.py --dataset TEaug-204 --data_size 384 --field 3.0 --out_vars PM --n_G_filters 36