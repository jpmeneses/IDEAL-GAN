#!/bin/bash

#SBATCH --job-name=v503-GEN
#SBATCH --output=out_503_GEN.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python test-genData.py --experiment_dir output/GAN-503 --n_samples 100