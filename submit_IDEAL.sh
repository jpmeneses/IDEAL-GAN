#!/bin/bash

#SBATCH --job-name=v108-unsup
#SBATCH --output=out_unsup_108.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-IDEAL-unsup.py --dataset Unsup-108 --rand_ne False --UQ True --epochs 60 --FM_TV_weight 1e-6