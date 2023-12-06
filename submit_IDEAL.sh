#!/bin/bash

#SBATCH --job-name=v205-TEaug
#SBATCH --output=out_TEaug_205.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-TEaug.py --dataset TEaug-205 --data_size 384 --DL_gen True --DL_experiment_dir output/GAN-503 --out_vars PM --n_G_filters 36