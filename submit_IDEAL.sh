#!/bin/bash

#SBATCH --job-name=v003-unsup
#SBATCH --output=out_unsup_003.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-unsup.py --dataset 'Unsup-003' --G_model 'encod-decod' --n_G_filters 32 --batch_size 1 --epochs 40 --epoch_decay 40 --epoch_ckpt 5 --lr 0.0001 --R2_TV_weight 0.0001