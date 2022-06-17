#!/bin/bash

#SBATCH --job-name=v004-IDEALGAN
#SBATCH --output=out_IDEALGAN_004.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL.py --dataset 'IDEAL-GAN-004' --G_model 'encod-decod' --te_input True --acqs_DataAug True --n_G_filters 32 --n_D_filters 32 --batch_size 4 --epochs 40 --epoch_decay 60 --epoch_ckpt 5 --lr 0.0002 --cycle_loss_weight 100.0 --B2A2B_weight 1.0 --R2_TV_weight 0.00001