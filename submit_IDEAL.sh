#!/bin/bash

#SBATCH --job-name=v007-VAE
#SBATCH --output=out_VAE_007.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-VAE.py --dataset VAE-007 --encoded_size 24 --adv_train True --batch_size 64 --lr 0.001 --D_lr_factor 5 --A_loss_weight 1e-2