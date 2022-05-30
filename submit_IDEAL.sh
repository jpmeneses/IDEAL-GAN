#!/bin/bash

#SBATCH --job-name=v006-IDEALGAN
#SBATCH --output=out_IDEALGAN_006.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL.py --dataset 'IDEAL-GAN-006' --G_model 'complex' --n_G_filters 32 --n_D_filters 64 --batch_size 4 --epochs 50 --epoch_decay 50 --lr 0.0001 --cycle_loss_weight 100.0 --B2A2B_weight 1.0 --R2_TV_weight 0.0001 --FM_TV_weight 0.0001