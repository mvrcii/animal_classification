#!/bin/bash
#SBATCH -p ls6
#SBATCH -J train_ac
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH -o logs/slurm-%j.out

python3 cross_validation.py --data_dir data --CV_fold_path data/cross_folds --model_name efficientnet_b3
