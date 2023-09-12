#!/bin/bash

#SBATCH --job-name=v102-TEaug
#SBATCH --output=out_TEaug_102.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-TEaug.py --dataset 'TEaug-102' --field 1.5 --out_vars 'PM' --epochs 100 --epoch_ckpt 20