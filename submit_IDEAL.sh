#!/bin/bash

#SBATCH --job-name=v308-sup
#SBATCH --output=out_sup_308.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-sup.py --dataset Sup-308 --data_size 384 --DL_gen True --DL_filename LDM_ds_noMEBCRN --G_model U-Net --batch_size 16 --epochs 50 --epoch_decay 50 --lr 0.001