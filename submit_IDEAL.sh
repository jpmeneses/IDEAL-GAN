#!/bin/bash

#SBATCH --job-name=v307-DS
#SBATCH --output=out_DS_307.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python gen_LDM_dataset.py --experiment_dir output/GAN-807 --ds_filename 'LDM_ds_noMEBCRN' --MEBCRN False --batch_size 10 --n_samples 3330