#!/bin/bash

#SBATCH --job-name=v113a-unsup
#SBATCH --output=out_unsup_113a.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-IDEAL-unsup.py --dataset Unsup-113a --rand_ne False --out_vars R2s --UQ True --UQ_R2s True --UQ_calib True --epochs 35 --epoch_ckpt 1 --lr 0.00001 --beta_1 0.5 --beta_2 0.9