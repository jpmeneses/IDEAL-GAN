#!/bin/bash

#SBATCH --job-name=v408d-GAN
#SBATCH --output=out_GAN_408d.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-ldm.py --experiment_dir 'GAN-408d' --n_timesteps 500 --beta_start 0.002 --beta_end 0.1 --batch_size 8 --epochs_ldm 2000 --epoch_ldm_ckpt 100 --lr 0.00015