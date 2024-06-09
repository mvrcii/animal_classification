import argparse
import os

import timm
import torch
from model.pytorch.callbacks import ModelCheckpoint
from model.pytorch.loggers import WandbLogger
from timm.data import create_transform

import wandb
from model import Trainer
from utils import setup_dataloaders, setup_reproducability, get_batch_size, init_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model_name', type=str, default='efficientnet_b3')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float, default=5e-4)
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

    wandb.init(project="animal_classifcation", entity="mvrcii_")
    wandb.run.name = os.path.basename(__file__)[:-3] + "_" + wandb.run.name.split("-")[2]
    wandb_logger = WandbLogger(experiment=wandb.run)

    model = init_model(model_name, num_classes)
    data_config = timm.data.resolve_model_data_config(model)

    train_loader, val_loader, test_loader = setup_dataloaders(data_config=data_config,
                                                              data_dir=data_dir,
                                                              seed=seed,
                                                              batch_size=batch_size,
                                                              num_workers=num_workers,
                                                              label_map=label_map)

    model = ImageClassifier(model=model)

    ckpt_callback = ModelCheckpoint(
        monitor="val_acc",
        mode='max',
        dirpath="weights",
        filename="best_" + model_name + '.pth.tar',
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

    trainer.fit(model, train_loader, val_loader)

    trainer.test(dataloaders=test_loader)
