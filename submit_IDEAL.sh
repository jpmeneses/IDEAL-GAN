#!/bin/bash

#SBATCH --job-name=v408-LDM
#SBATCH --output=out_LDM_408.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-ldm.py --experiment_dir 'GAN-408' --n_timesteps 1000 --beta_start 0.0015 --beta_end 0.0195 --batch_size 8 --epochs_ldm 1000 --lr 1e-5