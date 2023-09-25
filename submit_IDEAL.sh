#!/bin/bash

#SBATCH --job-name=v107-TEaug
#SBATCH --output=out_TEaug_107.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-TEaug.py --dataset 'TEaug-107' --DL_gen True --out_vars 'PM' --epochs 100 --epoch_ckpt 20