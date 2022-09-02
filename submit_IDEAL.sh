#!/bin/bash

#SBATCH --job-name=v005-unsup
#SBATCH --output=out_unsup_005.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-005' --UQ True --n_G_filters 32 --epochs 40 --epoch_decay 40 --epoch_ckpt 5 --lr 0.0001 --beta_1 0.9 --beta_2 0.999 --std_log_weight 0.0001