#!/bin/bash

#SBATCH --job-name=v012-sup
#SBATCH --output=out_sup_012.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-sup.py --dataset 'Sup-012' --out_vars 'WF' --G_model 'U-Net' --batch_size 1 --epoch_ckpt 20 --lr 0.0001 --D1_SelfAttention False