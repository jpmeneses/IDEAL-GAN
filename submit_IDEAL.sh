#!/bin/bash

#SBATCH --job-name=v001-IDEALGAN
#SBATCH --output=out_IDEALGAN_001.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL.py --dataset 'IDEAL-GAN-001' --FM_D_model 'PatchGAN' --R2_D_model 'PatchGAN' --n_G_filters 32 --n_D_filters 64 --batch_size 4 --epochs 50 --epoch_decay 50 --lr 0.0001 --D_R2_nds 1 --D_FM_nds 3 --cycle_loss_weight 100.0 --B2A2B_weight 0.5 --R2_critic_weight 1.0