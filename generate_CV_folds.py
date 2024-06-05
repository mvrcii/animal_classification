import argparse
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold


def generate_cross_validation_folds(data_dir, target_dir, n_splits, val_split_ratio=0.2):
    # Load train features and labels
    train_features = np.load(os.path.join(data_dir, 'train_features.npy'))
    train_labels = np.load(os.path.join(data_dir, 'train_labels.npy'))

    print(f"Loaded {len(train_features)} training samples and {len(train_labels)} labels.")

    # Initialize StratifiedKFold splitter
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Generate cross-validation folds
    fold_indices = []
    for fold_idx, (train_index, val_index) in enumerate(skf.split(train_features, train_labels)):
        # Split the train_index into train and validation sets based on val_split_ratio
        num_train_samples = int(len(train_index) * (1 - val_split_ratio))
        train_indices = train_index[:num_train_samples]
        val_indices = train_index[num_train_samples:]

        fold_indices.append((train_indices, val_indices))

        # Print fold statistics
        num_train_samples_fold = len(train_indices)
        num_val_samples_fold = len(val_indices)
        print(
            f"Fold {fold_idx + 1} - Train samples: {num_train_samples_fold}, Validation samples: {num_val_samples_fold}")

    # Create subdirectories for each fold
    for i, (train_indices, val_indices) in enumerate(fold_indices):
        fold_dir = os.path.join(target_dir, f'fold_{i}')
        os.makedirs(fold_dir, exist_ok=True)

        # Save fold indices
        np.save(os.path.join(fold_dir, 'train_indices.npy'), train_indices)
        np.save(os.path.join(fold_dir, 'val_indices.npy'), val_indices)

    print(
        f"Cross-validation folds with {n_splits} folds and validation split ratio {val_split_ratio} created successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate static cross-validation folds from data files.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the directory containing data files (train_features.npy and train_labels.npy).")
    parser.add_argument("--target_dir", type=str, required=True,
                        help="Path to the directory where fold indices will be saved.")
    parser.add_argument("--n_splits", type=int, default=10,
                        help="Number of folds for cross-validation.")
    parser.add_argument("--val_split_ratio", type=float, default=0.2,
                        help="Split ratio for validation data within each fold. Default is 0.2 (20%).")
    args = parser.parse_args()

    data_dir = args.data_dir
    target_dir = args.target_dir
    n_splits = args.n_splits
    val_split_ratio = args.val_split_ratio

    # Create target directory if it does not exist
    os.makedirs(target_dir, exist_ok=True)

    # Generate cross-validation folds with respective subdirectories
    generate_cross_validation_folds(data_dir, target_dir, n_splits, val_split_ratio)
