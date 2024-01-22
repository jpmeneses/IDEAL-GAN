#!/bin/bash

#SBATCH --job-name=v304-sup
#SBATCH --output=out_sup_304.txt
#SBATCH --partition=gpus
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jpmeneses@uc.cl

python train-sup.py --dataset Sup-304 --data_size 384 --DL_gen True --DL_LDM True --DL_experiment_dir output/GAN-716c --n_per_epoch 5000 --TE1 0.0014 --dTE 0.0022 --G_model U-Net --batch_size 16 --epochs 50 --epoch_decay 50