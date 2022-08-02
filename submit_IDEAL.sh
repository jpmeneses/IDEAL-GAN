#!/bin/bash

#SBATCH --job-name=v006-sup
#SBATCH --output=out_sup_006.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-sup.py --dataset 'Sup-006' --out_vars 'WF-PM' --G_model 'U-Net' --n_filters 72