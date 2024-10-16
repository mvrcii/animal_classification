import argparse
import os

import timm
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
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
    args = parse_args()
    try:
        if wandb.run:
            for key, value in vars(args).items():
                if key not in wandb.config:
                    setattr(wandb.config, key, value)
        return wandb.config
    except AttributeError:
        return args


def init_wandb(args):
    if 'WANDB_SWEEP_ID' in os.environ:
        # If there's a sweep ID, initialize WandB with argparse values as defaults
        wandb.init(project="animal_classifcation", config=vars(args))
    else:
        # Standard initialization with fixed configuration
        wandb.init(project="animal_classifcation", entity="mvrcii_", config=vars(args))

    # Set run name after initialization
    if wandb.run.sweep_id:
        sweep_id = wandb.run.sweep_id
        wandb.run.name = os.path.basename(__file__)[:-3] + "_sweep_" + sweep_id
    else:
        wandb.run.name = os.path.basename(__file__)[:-3] + "_single_" + wandb.run.id
    return wandb


def get_device():
    return "mps" if torch.backends.mps.is_available() else ("gpu" if torch.cuda.is_available() else "cpu")


def get_checkpoint_dir(wandb):
    base_dir = "weights"
    if 'WANDB_SWEEP_ID' in os.environ:
        sweep_id = wandb.run.sweep_id
        sweep_dir = os.path.join(base_dir, sweep_id)
        os.makedirs(sweep_dir, exist_ok=True)
        return sweep_dir
    else:
        os.makedirs(base_dir, exist_ok=True)
        return base_dir


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


def get_callbacks(model_name, id, wandb):
    ckpt_dir = get_checkpoint_dir(wandb)

    ckpt_callback = ModelCheckpoint(
        monitor="val_f1",
        mode='max',
        dirpath=ckpt_dir,
        filename=f"best_{model_name}_" + "epoch={epoch:02d}_valF1={val_f1:.4f}",
        save_top_k=1,
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val_f1',
        patience=10,
        strict=False,
        verbose=True,
        mode='max'
    )

    lr_callback = LearningRateMonitor(logging_interval='epoch')
    return [ckpt_callback, early_stop_callback, lr_callback]


def main():
    args = parse_args()
    wandb = init_wandb(args)
    config = wandb.config

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

    setup_reproducability(seed)

    fold_dir = get_fold_dir(config)
    device = get_device()

    callbacks = get_callbacks(model_name, id, wandb)

    trainer = Trainer(
        logger=wandb_logger,
        devices=1,
        accelerator=device,
        callbacks=callbacks,
        max_epochs=epochs,
    )

    model = ImageClassifier(
        model_name=model_name,
        lr=lr,
        num_classes=num_classes
    )

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


if __name__ == '__main__':
    main()
