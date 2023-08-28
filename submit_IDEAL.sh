#!/bin/bash

#SBATCH --job-name=v209-FID
#SBATCH --output=out_FID_209.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python test-genMetrics.py --experiment_dir 'output/GAN-209'