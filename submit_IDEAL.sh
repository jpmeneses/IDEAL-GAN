#!/bin/bash

#SBATCH --job-name=v104-GAN
#SBATCH --output=out_GAN_104.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-ldm.py --experiment_dir 'GAN-104' --n_timesteps 1000 --epoch_ldm_ckpt 5 --data_augmentation False