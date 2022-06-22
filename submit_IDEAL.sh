#!/bin/bash

#SBATCH --job-name=v000-TEaug
#SBATCH --output=out_TEaug_000.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-TEaug.py --dataset 'TEaug-000' --G_model 'encod-decod' --n_G_filters 32 --batch_size 1 --epochs 50 --epoch_decay 50 --epoch_ckpt 5 --lr 0.0001