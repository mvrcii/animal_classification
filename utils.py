import os
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from timm.data import create_transform
from torch.utils.data import DataLoader

from cross_validation import AnimalDataset


def load_files(root_dir):
    # Load train features and labels
    train_feat_path = os.path.join(root_dir, 'train_features.npy')
    train_label_path = os.path.join(root_dir, 'train_labels.npy')
    if os.path.exists(train_feat_path) and os.path.exists(train_label_path):
        train_features = np.load(train_feat_path)
        train_labels = np.load(train_label_path)
    else:
        raise ValueError("Train features and labels not found")

    # Load test features
    test_feat_path = os.path.join(root_dir, 'test_features.npy')
    if os.path.exists(test_feat_path):
        test_features = np.load(test_feat_path)
    else:
        raise ValueError("Test features not found")

    return train_features, train_labels, test_features


def setup_dataloaders(data_config, data_dir, seed, batch_size, num_workers, label_map, fold_dir=None):
    train_transform = create_transform(**data_config, is_training=True)
    val_transform = create_transform(**data_config)

    # Load the data
    train_features, train_labels, test_features = load_files(root_dir=data_dir)

    if fold_dir:
        # Load train and validation indices from fold files
        train_index = np.load(os.path.join(fold_dir, 'train_indices.npy'))
        val_index = np.load(os.path.join(fold_dir, 'val_indices.npy'))

        train_features = train_features[train_index]
        val_features = train_features[val_index]
        train_labels = train_labels[train_index]
        val_labels = train_labels[val_index]
    else:
        # Split train data into train and validation sets
        train_features, val_features, train_labels, val_labels = train_test_split(
            train_features, train_labels, test_size=0.2, random_state=seed
        )

    train_dataset = AnimalDataset(features=train_features, labels=train_labels, label_map=label_map,
                                  transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                              persistent_workers=True, pin_memory=True)

    val_dataset = AnimalDataset(features=val_features, labels=val_labels, label_map=label_map, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                            persistent_workers=True, pin_memory=True)

    test_dataset = AnimalDataset(features=test_features, label_map=label_map, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                             persistent_workers=True, pin_memory=True)

    return train_loader, val_loader, test_loader


def setup_reproducability(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
