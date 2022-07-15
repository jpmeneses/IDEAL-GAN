#!/bin/bash

#SBATCH --job-name=v004-sup
#SBATCH --output=out_sup_004.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-sup.py --dataset 'Sup-005' --out_vars 'WF-PM' --G_model 'multi-decod' --n_filters 72