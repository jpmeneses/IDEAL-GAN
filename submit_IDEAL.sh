#!/bin/bash

#SBATCH --job-name=v200-TEaug
#SBATCH --output=out_TEaug_200.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-TEaug.py --dataset 'TEaug-200' --data_size 384 --out_vars PM