#!/bin/bash

#SBATCH --job-name=v006-VAE
#SBATCH --output=out_VAE_006.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-VAE.py --dataset VAE-006 --encoded_size 24 --adv_train True --A_loss_weight 1e-8 --ls_reg_weight 2e-7