import argparse
import os

import numpy as np
import timm
import torch
from adabelief_pytorch import AdaBelief
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from timm.data import create_transform
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

import wandb
from dataset.animal_dataset import AnimalDataset
from utils import setup_reproducability


class ImageClassifier(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro')
        self.precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        img, label = batch
        y_pred = self(img)

        loss = self.loss(y_pred, label)
        acc = self.accuracy(y_pred, label)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch):
        imgs, labels = batch
        logits = self(imgs)

        loss = self.loss(logits, labels)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.accuracy(logits, labels), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_precision', self.precision(logits, labels), on_epoch=True, prog_bar=False)
        self.log('val_recall', self.recall(logits, labels), on_epoch=True, prog_bar=False)
        self.log('val_f1', self.f1(logits, labels), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdaBelief(self.parameters(), lr=lr, eps=1e-16, betas=(0.9, 0.999), weight_decouple=True,
                              rectify=False, weight_decay=2e-4)
        # return optim.AdamW(self.parameters(), lr=lr, weight_decay=2e-4)
        return optimizer


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


def setup_dataloaders(data_config, fold_dir=None):
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


def init_model(model_name, num_classes):
    if model_name == 'efficientnet_b0':
        return timm.create_model('efficientnet_b0.ra_in1k', pretrained=True, num_classes=num_classes)
    elif model_name == 'efficientnet_b3':
        return timm.create_model('efficientnet_b3.ra2_in1k', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError("Invalid model name")


def get_batch_size(model_name):
    if model_name == 'efficientnet_b0':
        return 64
    elif model_name == 'efficientnet_b3':
        return 16
    else:
        raise ValueError("Invalid model name")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--CV_fold_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='efficientnet_b0')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    for (k, v) in vars(args).items():
        print(k, ":", v)

    model_name = args.model_name
    data_dir = args.data_dir
    num_classes = 6
    batch_size = args.batch_size if args.batch_size else get_batch_size(model_name)
    lr = args.lr
    epochs = args.epochs
    seed = args.seed
    num_workers = args.num_workers
    CV_fold_path = args.CV_fold_path

    label_map = {
        'bird': 0,
        'cat': 1,
        'deer': 2,
        'dog': 3,
        'frog': 4,
        'horse': 5
    }

    setup_reproducability(seed)

    device = "mps" if torch.backends.mps.is_available() else ("gpu" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dirName = "weights"
    try:
        os.makedirs(dirName)
        print("Checkpoint directory", dirName, "created")
    except FileExistsError:
        print("Checkpoint directory", dirName, "already exists")

    group_id = wandb.util.generate_id()

    CV_fold_folders = [x for x in os.listdir(CV_fold_path) if x.startswith("fold")]
    CV_fold_folders = sorted(CV_fold_folders)
    number_of_experiments = len(CV_fold_folders)

    for i in range(number_of_experiments):
        id = i + 1
        print("\n--------------> Starting Fold " + str(id))

        wandb.init(project="animal_classifcation",
                   group=model_name + "_CV_C" + "_" + group_id,
                   save_code=True,
                   reinit=True)
        wandb.run.name = "fold_" + str(id)
        wandb.run.save()

        config = wandb.config
        config.exp = os.path.basename(__file__)[:-3]
        config.model = model_name
        config.dataset = "cv_fold_Dataset"
        config.lr = lr
        config.bs = batch_size
        config.num_workers = num_workers
        config.seed = seed

        wandb_logger = WandbLogger(experiment=wandb.run)

        fold_dir = os.path.join(CV_fold_path, CV_fold_folders[i])

        ckpt_callback = ModelCheckpoint(
            monitor="val_acc",
            mode='max',
            dirpath="weights",
            filename="best_" + model_name + '_' + str(id) + '.pth.tar',
            save_top_k=1,
            verbose=True
        )

        trainer = Trainer(
            logger=wandb_logger,
            devices=1,
            accelerator=device,
            callbacks=[ckpt_callback],
            max_epochs=epochs,
        )

        model = init_model(model_name, num_classes)

        data_config = timm.data.resolve_model_data_config(model)

        train_loader, val_loader, test_loader = setup_dataloaders(data_config=data_config, fold_dir=fold_dir)

        model = ImageClassifier(model=model)

        # >==== TRAINING ====<
        trainer.fit(model, train_loader, val_loader)

        # >==== TESTING ====<
        trainer.test(dataloaders=test_loader)

        wandb.finish()
