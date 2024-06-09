from adabelief_pytorch import AdaBelief
from lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

from utils import init_model


class ImageClassifier(LightningModule):
    def __init__(self, model_name, lr=1e-4, num_classes=6):
        super().__init__()
        self.lr = lr
        self.model = init_model(model_name, num_classes)

        self.loss = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro')
        self.precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

        self.save_hyperparameters()

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
        optimizer = AdaBelief(self.parameters(), lr=self.lr, eps=1e-16, betas=(0.9, 0.999), weight_decouple=True,
                              rectify=False, weight_decay=2e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
