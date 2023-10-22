#!/bin/bash

#SBATCH --job-name=v002-VAE
#SBATCH --output=out_VAE_002.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-VAE.py --dataset VAE-002 --encoded_size 24 --A_loss_weight 0.0001