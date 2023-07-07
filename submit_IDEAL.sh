#!/bin/bash

#SBATCH --job-name=v010-sup
#SBATCH --output=out_sup_010.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-sup.py --dataset 'Sup-010' --out_vars 'PM' --batch_size 1 --epoch_ckpt 20 --lr 0.0001