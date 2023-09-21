#!/bin/bash

#SBATCH --job-name=v102-sup
#SBATCH --output=out_sup_102.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-sup.py --dataset 'Sup-102' --out_vars 'WF-PM' --epoch_ckpt 20 --D2_SelfAttention False