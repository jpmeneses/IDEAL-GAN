#!/bin/bash

#SBATCH --job-name=v000-VAE
#SBATCH --output=out_VAE_000.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-VAE.py --dataset VAE-000