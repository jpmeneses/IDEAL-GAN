#!/bin/bash

#SBATCH --job-name=v101-sup
#SBATCH --output=out_sup_101.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-sup.py --dataset 'Sup-101' --n_echoes 3 --out_vars 'WF-PM' --epochs 120 --epoch_decay 100 --epoch_ckpt 20 --lr 0.0005