import argparse
import os

import timm
import torch
from adabelief_pytorch import AdaBelief
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from timm.data import create_transform
from torch import nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

import wandb
from utils import setup_dataloaders, setup_reproducability, get_batch_size


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

    def test_step(self, batch):
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
        return optimizer


def init_model(model_name, num_classes):
    if model_name == 'efficientnet_b0':
        return timm.create_model('efficientnet_b0.ra_in1k', pretrained=True, num_classes=num_classes)
    elif model_name == 'efficientnet_b3':
        return timm.create_model('efficientnet_b3.ra2_in1k', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError("Invalid model name")


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
