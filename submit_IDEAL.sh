#!/bin/bash

#SBATCH --job-name=v004-sup
#SBATCH --output=out_sup_004.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-sup.py --dataset 'Sup-004' --out_vars 'WF-PM' --G_model 'U-Net' --n_filters 72 --D1_SelfAttention True