#!/bin/bash

#SBATCH --job-name=v011-test
#SBATCH --output=out_test_011.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl	

python train-IDEAL-TEaug.py --dataset 'test-011' --G_model 'MEBCRN' --out_vars 'WFc' --te_input False --epoch_decay 200 --epoch_ckpt 50 --beta_1 0.9 --beta_2 0.999