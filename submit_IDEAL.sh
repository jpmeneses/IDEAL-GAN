#!/bin/bash

#SBATCH --job-name=v000-sup
#SBATCH --output=out_sup_000.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-sup.py --dataset 'Sup-000' --out_vars 'WF' --G_model 'U-Net' 