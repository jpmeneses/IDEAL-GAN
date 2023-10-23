#!/bin/bash

#SBATCH --job-name=v005-VAE
#SBATCH --output=out_VAE_005.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-VAE.py --dataset VAE-005 --encoded_size 24 --A_loss_weight 1e-7