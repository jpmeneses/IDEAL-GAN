#!/bin/bash

#SBATCH --job-name=v011-sup
#SBATCH --output=out_sup_011.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-sup.py --dataset 'Sup-011' --out_vars 'WF-PM' --batch_size 1 --epoch_ckpt 20 --lr 0.0001 --D2_SelfAttention False