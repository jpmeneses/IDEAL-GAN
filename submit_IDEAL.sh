#!/bin/bash

#SBATCH --job-name=v103-sup
#SBATCH --output=out_sup_103.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-sup.py --dataset 'Sup-103' --n_echoes 3 --out_vars 'PM' --epoch_ckpt 20