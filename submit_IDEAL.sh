#!/bin/bash

#SBATCH --job-name=v110-TEaug
#SBATCH --output=out_TEaug_110.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-TEaug.py --dataset 'TEaug-110' --DL_gen True --DL_experiment_dir 'output/GAN-407' --out_vars PM 