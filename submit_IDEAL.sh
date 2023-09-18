#!/bin/bash

#SBATCH --job-name=v106-TEaug
#SBATCH --output=out_TEaug_106.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-TEaug.py --dataset 'TEaug-106' --field 3.0 --out_vars 'PM' --epochs 100 --epoch_ckpt 20 --FM_aug True --FM_mean 2.0