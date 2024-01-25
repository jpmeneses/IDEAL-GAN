#!/bin/bash

#SBATCH --job-name=v307-sup
#SBATCH --output=out_sup_307.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-sup.py --dataset Sup-307 --data_size 384 --DL_gen True --DL_experiment_dir output/GAN-722b --n_per_epoch 5000 --DL_LDM True --DDIM True --TE1 0.0023 --dTE 0.0023 --G_model U-Net --batch_size 16 --epochs 50 --epoch_decay 50 --lr 0.001