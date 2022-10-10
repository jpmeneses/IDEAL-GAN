#!/bin/bash

#SBATCH --job-name=v007-unsup
#SBATCH --output=out_unsup_007.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-007' --R2_log True --epochs 25 --epoch_decay 25 --epoch_ckpt 5 --lr 0.0001 --beta_1 0.9 --beta_2 0.999 --R2_L1_weight 0.1