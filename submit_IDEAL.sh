#!/bin/bash

#SBATCH --job-name=v722b-LDM
#SBATCH --output=out_LDM_722b.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-ldm.py --experiment_dir output/GAN-722b --n_timesteps 500 --batch_size 8 --epochs_ldm 500 --epoch_ldm_ckpt 100 --lr 7e-5