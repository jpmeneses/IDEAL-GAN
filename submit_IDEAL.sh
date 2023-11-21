#!/bin/bash

#SBATCH --job-name=v408g-LDM
#SBATCH --output=out_LDM_408g.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-ldm.py --experiment_dir 'GAN-408g' --n_timesteps 1000 --beta_start 0.0005 --beta_end 0.05 --batch_size 8 --epochs_ldm 2000 --epoch_ldm_ckpt 100 --lr 0.00007