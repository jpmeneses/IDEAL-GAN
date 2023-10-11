#!/bin/bash

#SBATCH --job-name=v002-sGAN
#SBATCH --output=out_sGAN_002.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-singleGAN.py --dataset 'sinGAN-WF' --K_sc 3
python train-singleGAN.py --dataset 'sinGAN-WF' --K_sc 2
python train-singleGAN.py --dataset 'sinGAN-WF' --K_sc 1
python train-singleGAN.py --dataset 'sinGAN-WF' --K_sc 0