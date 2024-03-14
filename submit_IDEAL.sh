#!/bin/bash

#SBATCH --job-name=v200-sup
#SBATCH --output=out_sup_200.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-sup.py --dataset Sup-200 --data_size 384 --out_vars PM --n_G_filters 36 --lr 0.0001