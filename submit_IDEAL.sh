#!/bin/bash

#SBATCH --job-name=v004-VAE
#SBATCH --output=out_VAE_004.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-VAE.py --dataset VAE-004 --encoded_size 24 --A_loss_weight 1e-6