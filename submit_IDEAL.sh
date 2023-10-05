#!/bin/bash

#SBATCH --job-name=v999-sup
#SBATCH --output=out_sup_999.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-sup.py --dataset 'Sup-999' --out_vars 'PM' --n_G_filters 36 --batch_size 1 --epochs 20 --epoch_decay 20 --lr 0.0001