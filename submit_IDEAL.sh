#!/bin/bash

#SBATCH --job-name=v321-sup
#SBATCH --output=out_sup_321.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-sup.py --dataset Sup-321 --data_size 384 --G_model U-Net --batch_size 16 --lr 0.001