#!/bin/bash

#SBATCH --job-name=v005-TEaug
#SBATCH --output=out_TEaug_005.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-TEaug.py --dataset 'TEaug-005' --n_G_filters 32 --epochs 200 --epoch_decay 200 --epoch_ckpt 25 --lr 0.0001 --beta_1 0.9 --beta_2 0.999