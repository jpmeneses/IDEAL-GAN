#!/bin/bash

#SBATCH --job-name=v111-unsup
#SBATCH --output=out_unsup_111.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-IDEAL-unsup.py --dataset Unsup-111 --rand_ne False --out_vars PM --UQ True --UQ_R2s True --batch_size 8 --epochs 50 --epoch_ckpt 2 --lr 0.001