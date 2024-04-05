#!/bin/bash

#SBATCH --job-name=v316-sup
#SBATCH --output=out_sup_316.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-sup.py --dataset Sup-316 --data_size 384 --DL_gen True --DL_filename LDM_ds_noMEBCRN --TE1 0.0012 --dTE 0.0020 --G_model U-Net --batch_size 16 --lr 0.001