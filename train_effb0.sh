#!/bin/bash
#SBATCH -p ls6
#SBATCH -J train_ac
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH -o logs/slurm-%j.out

python3 train_classification_model.py --data_dir data
