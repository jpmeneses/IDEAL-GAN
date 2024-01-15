#!/bin/bash

#SBATCH --job-name=v302-sup
#SBATCH --output=out_sup_302.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-sup.py --dataset Sup-302 --data_size 384 --DL_gen True --DL_experiment_dir output/GAN-715 --n_per_epoch 5000 --TE1 0.0023 --dTE 0.0023 --G_model U-Net --epochs 60 --epoch_decay 60