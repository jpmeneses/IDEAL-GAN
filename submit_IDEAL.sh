#!/bin/bash

#SBATCH --job-name=v104-DDPM
#SBATCH --output=out_DDPM_104.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-ldm.py --experiment_dir 'GAN-104' --n_timesteps 200 --batch_size 8 --epoch_ldm_ckpt 5 --data_augmentation True --lr 1e-4