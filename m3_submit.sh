#!/bin/bash

#SBATCH --job-name=v000-Unsup
#SBATCH --account=dq13
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=m3h

python train-IDEAL-unsup.py --dataset 'Unsup-000' --UQ True --epochs 30 --epochs_decay 30