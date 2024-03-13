#!/bin/bash

#SBATCH --job-name=v202-sup
#SBATCH --output=out_sup_202.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-sup.py --dataset Sup-202 --data_size 384 --G_model U-Net --n_G_filters 36 --lr 0.0001