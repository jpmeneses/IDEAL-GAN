#!/bin/bash

#SBATCH --job-name=v259-DDPM
#SBATCH --output=out_DDPM_259.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-ldm.py --experiment_dir 'GAN-259' --n_timesteps 1000 --batch_size 16 --epochs_ldm 200 --lr 1e-4