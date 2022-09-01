#!/bin/bash

#SBATCH --job-name=v005-unsup
#SBATCH --output=out_unsup_005.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-005' --n_G_filters 64 --epochs 200 --epoch_decay 200 --epoch_ckpt 25 --lr 0.0001 --beta_1 0.9 --beta_2 0.999 --std_log_weight 0.01