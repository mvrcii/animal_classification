import argparse
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold


def generate_cross_validation_folds(data_dir, target_dir, n_splits):
    # Load train features and labels
    train_features = np.load(os.path.join(data_dir, 'train_features.npy'))
    train_labels = np.load(os.path.join(data_dir, 'train_labels.npy'))

    # Initialize StratifiedKFold splitter
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Generate cross-validation folds
    fold_indices = []
    for train_index, val_index in skf.split(train_features, train_labels):
        fold_indices.append((train_index, val_index))

    # Create subdirectories for each fold
    for i, (train_index, val_index) in enumerate(fold_indices):
        fold_dir = os.path.join(target_dir, f'fold_{i}')
        os.makedirs(fold_dir, exist_ok=True)

        # Save fold indices
        np.save(os.path.join(fold_dir, 'train_indices.npy'), train_index)
        np.save(os.path.join(fold_dir, 'val_indices.npy'), val_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate static cross-validation folds from data files.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the directory containing data files (train_features.npy and train_labels.npy).")
    parser.add_argument("--target_dir", type=str, required=True,
                        help="Path to the directory where fold indices will be saved.")
    parser.add_argument("--n_splits", type=int, default=10,
                        help="Number of folds for cross-validation.")
    args = parser.parse_args()

    data_dir = args.data_dir
    target_dir = args.target_dir
    n_splits = args.n_splits

    # Create target directory if it does not exist
    os.makedirs(target_dir, exist_ok=True)

    # Generate cross-validation folds with respective subdirectories
    generate_cross_validation_folds(data_dir, target_dir, n_splits)
