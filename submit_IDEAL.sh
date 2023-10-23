#!/bin/bash

#SBATCH --job-name=v258-GAN
#SBATCH --output=out_GAN_258.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-GAN.py --dataset 'GAN-258' --PM_bayes_layer True --encoded_size 24 --adv_train True --cGAN True --A_loss_weight 1e-5 --B_loss_weight 0.05 --FM_loss_weight 1.0