#!/bin/bash

#SBATCH --job-name=v309-sup
#SBATCH --output=out_sup_309.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-sup.py --dataset Sup-309 --data_size 384 --DL_gen True --DL_filename LDM_ds_noMEBCRN --sigma_noise 0.05 --G_model U-Net --batch_size 16 --epochs 50 --epoch_decay 50 --lr 0.001