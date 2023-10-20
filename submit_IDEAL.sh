#!/bin/bash

#SBATCH --job-name=v001-VAE
#SBATCH --output=out_VAE_001.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-VAE.py --dataset VAE-001 --encoded_size 24 --adv_train True