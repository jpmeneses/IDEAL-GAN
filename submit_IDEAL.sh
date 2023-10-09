#!/bin/bash

#SBATCH --job-name=v001-sGAN
#SBATCH --output=out_sGAN_001.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-singleGAN.py --dataset 'sinGAN-wMsk-v2' --K_sc 3 --DC_loss_weight 50.0
python train-singleGAN.py --dataset 'sinGAN-wMsk-v2' --K_sc 2 --DC_loss_weight 50.0
python train-singleGAN.py --dataset 'sinGAN-wMsk-v2' --K_sc 1 --DC_loss_weight 50.0
python train-singleGAN.py --dataset 'sinGAN-wMsk-v2' --K_sc 0 --DC_loss_weight 50.0