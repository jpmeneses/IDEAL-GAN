#!/bin/bash

#SBATCH --job-name=v317-sup
#SBATCH --output=out_sup_317.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-sup.py --dataset Sup-317 --data_size 384 --DL_gen True --DL_partial_real True --DL_filename LDM_ds_noMEBCRN --G_model U-Net --batch_size 16 --lr 0.001