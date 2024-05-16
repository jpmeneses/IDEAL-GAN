#!/bin/bash

#SBATCH --job-name=v322-sup
#SBATCH --output=out_sup_322.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-sup.py --dataset Sup-322 --data_size 384 --DL_partial_real False --G_model U-Net --batch_size 16 --lr 0.001