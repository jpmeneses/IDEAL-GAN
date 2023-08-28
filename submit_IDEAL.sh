#!/bin/bash

#SBATCH --job-name=v207-FID
#SBATCH --output=out_FID_207.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python test-genMetrics.py --experiment_dir 'output/GAN-207'