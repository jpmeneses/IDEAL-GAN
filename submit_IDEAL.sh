#!/bin/bash

#SBATCH --job-name=v408-LDM
#SBATCH --output=out_LDM_408.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-ldm.py --experiment_dir 'GAN-408' --n_timesteps 200 --batch_size 8 --epochs_ldm 800 --lr 5e-5