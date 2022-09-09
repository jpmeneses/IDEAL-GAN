#!/bin/bash

#SBATCH --job-name=v008-sup
#SBATCH --output=out_sup_008.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-sup.py --dataset 'Sup-008' --out_vars 'WF' --n_filters 72 --batch_size 1 --epochs 200 --epoch_decay 200 --epoch_ckpt 25 --lr 0.0001 --beta_1 0.9 --beta_2 0.999