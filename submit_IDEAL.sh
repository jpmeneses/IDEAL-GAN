#!/bin/bash

#SBATCH --job-name=v105-sup
#SBATCH --output=out_sup_105.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-sup.py --dataset 'Sup-105' --n_echoes 3 --out_vars 'WF-PM' --epoch_ckpt 20 --D2_SelfAttention False