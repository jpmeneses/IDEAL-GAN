#!/bin/bash

#SBATCH --job-name=v203-sup
#SBATCH --output=out_sup_203.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-sup.py --dataset Sup-203 --n_echoes 3 --out_vars 'WF' --G_model 'U-Net' --n_G_filters 36 --batch_size 1 --epoch_ckpt 20 --lr 0.0001