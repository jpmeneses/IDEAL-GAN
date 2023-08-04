#!/bin/bash

#SBATCH --job-name=v102-DDPM
#SBATCH --output=out_DDPM_102.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-ldm.py --experiment_dir 'GAN-102' --n_timesteps 1000 --epoch_ldm_ckpt 5 --data_augmentation False