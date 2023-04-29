#!/bin/bash

#SBATCH --job-name=v104-sup
#SBATCH --output=out_sup_104.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-sup.py --dataset 'Sup-104' --out_vars 'WFc' --G_model 'MEBCRN' --n_G_filters 64 --epochs 200 --epoch_decay 200 --epoch_ckpt 50 --beta_1 0.9 --beta_2 0.999