#!/bin/bash

#SBATCH --job-name=v327-sup
#SBATCH --output=out_sup_327.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-sup.py --dataset Sup-327 --data_size 384 --DL_gen True --DL_filename LDM_ds_3330 --TE1 0.0012 --dTE 0.0018 --G_model U-Net --n_G_filters 36 --batch_size 16