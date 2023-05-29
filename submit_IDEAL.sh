#!/bin/bash

#SBATCH --job-name=t006-GAN
#SBATCH --output=test_GAN_006.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python test-genData.py --experiment_dir 'output/GAN-006'