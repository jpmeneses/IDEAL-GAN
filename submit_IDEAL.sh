#!/bin/bash

#SBATCH --job-name=v409-LDM
#SBATCH --output=out_LDM_409.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-ldm.py --experiment_dir 'GAN-409' --n_timesteps 300 --beta_start 0.005 --beta_end 0.15 --batch_size 8 --epochs_ldm 600 --lr 1e-5