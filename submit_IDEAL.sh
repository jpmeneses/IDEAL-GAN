#!/bin/bash

#SBATCH --job-name=v010-TEaug
#SBATCH --output=out_TEaug_010.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-TEaug.py --dataset 'TEaug-010' --G_model 'multi-decod' --out_vars 'WF-PM' --te_input False --epoch_decay 200 --epoch_ckpt 50 --beta_1 0.9 --beta_2 0.999 --D2_SelfAttention False