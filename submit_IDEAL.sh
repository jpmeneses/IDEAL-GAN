#!/bin/bash

#SBATCH --job-name=v104-sup
#SBATCH --output=out_sup_104.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-sup.py --dataset 'Sup-104' --n_echoes 3 --out_vars 'WF' --G_model 'U-Net' --epoch_ckpt 20