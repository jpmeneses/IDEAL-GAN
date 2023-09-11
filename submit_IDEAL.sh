#!/bin/bash

#SBATCH --job-name=v100-TEaug
#SBATCH --output=out_TEaug_100.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-TEaug.py --dataset 'TEaug-100' --field 1.5 --out_vars 'PM' --epochs 200 --epoch_ckpt 20