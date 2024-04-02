#!/bin/bash

#SBATCH --job-name=v311-sup
#SBATCH --output=out_sup_311.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-sup.py --dataset Sup-311 --data_size 384 --DL_gen True --DL_filename LDM_ds_noMEBCRN --sigma_noise 0.02 --G_model U-Net --batch_size 16 --lr 0.001