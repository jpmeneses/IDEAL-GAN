#!/bin/bash

#SBATCH --job-name=v507-GAN
#SBATCH --output=out_GAN_507.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-IDEAL-GAN.py --dataset GAN-507 --data_size 384 --rand_ne True --n_downsamplings 4 --div_decod True  --encoded_size 3 --ls_mean_activ None --adv_train True --cGAN True  --batch_size 2 --epochs 140 --epoch_decay 140 --lr 0.00025 --ls_reg_weight 1e-6
