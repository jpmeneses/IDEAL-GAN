#!/bin/bash

#SBATCH --job-name=v101-DDPM
#SBATCH --output=out_DDPM_101.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-ldm.py --experiment_dir 'GAN-101' --epoch_ldm_ckpt 5 --data_augmentation True