import argparse
import os

import timm
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from timm.data import create_transform

import wandb
from classifier_module import ImageClassifier
from utils import setup_dataloaders, setup_reproducability, get_batch_size


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--CV_fold_path', type=str, default='data/cross_folds')
    parser.add_argument('--fold_id', type=int, help="Fold id to train on from 0 to 9.")
    parser.add_argument('--model_name', type=str, default='efficientnet_b3')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=8)
    return parser.parse_args()


def get_config():
    try:
        # Try using WandB config if available
        import wandb
        config = wandb.config
    except (ImportError, AttributeError):
        # Fallback to argparse if WandB is not available or not initialized
        config = parse_args()
    return config


def init_wandb():
    if 'WANDB_SWEEP_ID' in os.environ:
        # If there's a sweep ID, it means we are running as part of a sweep
        wandb.init(project="animal_classification", config=get_config())
    else:
        # Standard initialization with fixed configuration
        wandb.init(project="animal_classification", entity="mvrcii_", config=get_config())

    # Update run name based on WandB initialization
    wandb.run.name = os.path.basename(__file__)[:-3] + "_" + wandb.run.name.split("-")[2]
    return wandb


def get_fold_dir(config):
    CV_fold_path = config.CV_fold_path
    fold_id = config.fold_id
    fold_dir = None
    if CV_fold_path is not None and fold_id is not None:
        CV_fold_folders = [x for x in os.listdir(CV_fold_path) if x.startswith("fold")]
        CV_fold_folders = sorted(CV_fold_folders)
        fold_dir = os.path.join(CV_fold_path, CV_fold_folders[fold_id])
        print("Training on fold id:", fold_id)
        print("Fold directory:", fold_dir)
    else:
        print("Fold id not provided. Training on random train-val split.")
    return fold_dir


if __name__ == '__main__':
    config = get_config()
    wandb = init_wandb()
    wandb_logger = WandbLogger(experiment=wandb.run)

    model_name = config.model_name
    data_dir = config.data_dir
    num_classes = 6
    config.batch_size = config.batch_size if config.batch_size else get_batch_size(model_name)
    batch_size = config.batch_size
    lr = config.lr
    epochs = config.epochs
    seed = config.seed
    num_workers = config.num_workers

    label_map = {
        'bird': 0,
        'cat': 1,
        'deer': 2,
        'dog': 3,
        'frog': 4,
        'horse': 5
    }

    for (k, v) in vars(config).items():
        print(k, ":", v)

    fold_dir = get_fold_dir(config)
    setup_reproducability(seed)

    device = "mps" if torch.backends.mps.is_available() else ("gpu" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dirName = "weights"
    os.makedirs(dirName, exist_ok=True)

    ckpt_callback = ModelCheckpoint(
        monitor="val_f1",
        mode='max',
        dirpath=os.path.join("weights"),
        filename=f"best_{model_name}_{id}_" + "epoch={epoch:02d}_valF1={val_f1:.4f}",
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

    model = ImageClassifier(model_name=model_name, num_classes=num_classes)

    data_config = timm.data.resolve_model_data_config(model)

    train_loader, val_loader, test_loader = setup_dataloaders(data_config=data_config,
                                                              data_dir=data_dir,
                                                              fold_dir=fold_dir,
                                                              seed=seed,
                                                              batch_size=batch_size,
                                                              num_workers=num_workers,
                                                              label_map=label_map)

    trainer.fit(model, train_loader, val_loader)

    trainer.test(dataloaders=test_loader, ckpt_path='best')

    wandb.finish()
