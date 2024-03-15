#!/bin/bash

#SBATCH --job-name=v806-GAN
#SBATCH --output=out_GAN_806.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-IDEAL-GAN.py --dataset GAN-806 --data_size 384 --rand_ne True --rand_ph_offset False --only_mag True --n_G_filt_list 36,72,72,144 --n_downsamplings 3 --div_decod True  --encoded_size 3 --adv_train True --cGAN True --batch_size 2 --epochs 200 --epoch_decay 200 --epoch_ckpt 50 --lr 0.001 --FM_loss_weight 0.5 --ls_reg_weight 1e-6