#!/bin/bash

#SBATCH --job-name=v012-TEaug
#SBATCH --output=out_TEaug_012.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-TEaug.py --dataset 'TEaug-012' --out_vars 'PM' --epoch_decay 200 --epoch_ckpt 50 --beta_1 0.9 --beta_2 0.999