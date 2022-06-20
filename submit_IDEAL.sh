#!/bin/bash

#SBATCH --job-name=v001-TEaug
#SBATCH --output=out_TEaug_001.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-TEaug.py --dataset 'TEaug-001' --G_model 'encod-decod' --n_G_filters 32 --batch_size 4 --epochs 40 --epoch_decay 40 --epoch_ckpt 5 --lr 0.0002