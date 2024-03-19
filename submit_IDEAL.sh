#!/bin/bash

#SBATCH --job-name=v806a-LDM
#SBATCH --output=out_LDM_806a.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-ldm.py --experiment_dir output/GAN-806a --n_timesteps 500 --batch_size 8 --epochs_ldm 300 --epoch_ldm_ckpt 100 --lr 7e-5