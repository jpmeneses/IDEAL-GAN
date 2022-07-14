#!/bin/bash

#SBATCH --job-name=v003-sup
#SBATCH --output=out_sup_003.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL.py --dataset 'Sup-003' --G_model 'multi-decod' --n_filters 72