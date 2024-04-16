#!/bin/bash

#SBATCH --job-name=v320-sup
#SBATCH --output=out_sup_320.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-sup.py --dataset Sup-320 --data_size 384 --DL_gen True --DL_filename LDM_ds_noMEBCRN_6660 --sigma_noise 0.02 --shuffle False --TE1 0.0014 --dTE 0.0022 --G_model U-Net --batch_size 16 --lr 0.001