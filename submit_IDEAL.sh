#!/bin/bash

#SBATCH --job-name=v325-sup
#SBATCH --output=out_sup_325.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python gen_LDM_dataset.py --experiment_dir output/GAN-813 --MEBCRN False --batch_size 10 --n_samples 3130
python train-sup.py --dataset Sup-325 --data_size 384 --DL_gen True --DL_partial_real True --DL_filename LDM_ds_3130 --G_model U-Net --n_G_filters 36 --batch_size 16