#!/bin/bash

#SBATCH --job-name=v004-unsup
#SBATCH --output=out_unsup_004.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-004' --UQ True --epochs 30 --epoch_decay 30 --lr 0.0002 --beta_1 0.9 --beta_2 0.999 --std_log_weight 1e-2 --R2_L1_weight 1e3