#!/bin/bash

#SBATCH --job-name=v716-LDM
#SBATCH --output=out_LDM_716.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-ldm.py --experiment_dir output/GAN-716 --scheduler cosine --n_timesteps 1000 --batch_size 8 --epochs_ldm 2000 --epoch_ldm_ckpt 100 --lr 7e-5