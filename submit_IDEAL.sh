#!/bin/bash

#SBATCH --job-name=v300-sup
#SBATCH --output=out_sup_300.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-sup.py --dataset Sup-300 --data_size 384 --DL_gen True --DL_experiment_dir output/GAN-715 --n_per_epoch 5000 --G_model U-Net