import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class AnimalDataset(Dataset):
    def __init__(self, features, label_map, labels=None, transform=None):
        if features is None:
            raise ValueError("Features cannot be None")
        self.features = features
        self.labels = labels
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        img = self.features[idx]

        # Repeat single-channel image array along a new dimension to create RGB image
        img_rgb = np.repeat(img[..., np.newaxis], 3, -1)
        img_rgb = np.squeeze(img_rgb, axis=-2)

        # Convert numpy array to PIL Image
        img = Image.fromarray(img_rgb.astype('uint8'))

        if self.transform:
            img = self.transform(img)

        if self.labels is not None:
            label = self.labels[idx]
            # Map label to numerical value
            label = self.label_map[label]
            return img, label
        else:
            return img
