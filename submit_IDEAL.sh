#!/bin/bash

#SBATCH --job-name=v001-sup
#SBATCH --output=out_sup_001.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-sup.py --dataset 'Sup-002' --out_vars 'PM' --G_model 'U-Net' 