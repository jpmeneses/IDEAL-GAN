#!/bin/bash

#SBATCH --job-name=v600-LDM
#SBATCH --output=out_600_LDM.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-ldm.py --experiment_dir GAN-600 --scheduler cosine --n_timesteps 1000 --batch_size 8 --epochs_ldm 2 --epoch_ldm_ckpt 1
python train-ldm.py --experiment_dir GAN-601 --scheduler cosine --n_timesteps 1000 --batch_size 8 --epochs_ldm 2 --epoch_ldm_ckpt 1