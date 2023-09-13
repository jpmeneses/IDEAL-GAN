#!/bin/bash

#SBATCH --job-name=v104-TEaug
#SBATCH --output=out_TEaug_104.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-TEaug.py --dataset 'TEaug-104' --out_vars 'PM' --epochs 100 --epoch_ckpt 20