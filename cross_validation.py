import argparse
import json
import os

import numpy as np
import timm
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from timm.data import create_transform
from wandb.util import generate_id

import wandb
from classifier_module import ImageClassifier
from utils import setup_reproducability, setup_dataloaders, get_batch_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--CV_fold_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='CV_results')
    parser.add_argument('--model_name', type=str, default='efficientnet_b3')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=200)
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
    output_dir = args.output_dir

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

    group_id = generate_id()

    CV_fold_folders = [x for x in os.listdir(CV_fold_path) if x.startswith("fold")]
    CV_fold_folders = sorted(CV_fold_folders)
    number_of_experiments = len(CV_fold_folders)

    all_predictions = []
    val_results_dict = {}

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
            monitor="val_f1",
            mode='max',
            dirpath=os.path.join("weights", group_id),
            filename=f"best_{model_name}_{id}_" + "epoch={epoch:02d}_valF1={val_f1:.4f}",
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

        trainer = Trainer(
            logger=wandb_logger,
            devices=1,
            accelerator=device,
            callbacks=[ckpt_callback, early_stop_callback, lr_callback],
            max_epochs=epochs,
        )

        model = ImageClassifier(model_name=model_name, num_classes=num_classes, lr=lr)

        data_config = timm.data.resolve_model_data_config(model)

        train_loader, val_loader, test_loader = setup_dataloaders(data_config=data_config,
                                                                  fold_dir=fold_dir,
                                                                  data_dir=data_dir,
                                                                  seed=seed,
                                                                  batch_size=batch_size,
                                                                  num_workers=num_workers,
                                                                  label_map=label_map)

        # >==== TRAINING ====<
        trainer.fit(model, train_loader, val_loader)

        # >==== TESTING ====<
        print(">==== EVALUATING & PREDICTING ====<")
        val_results = trainer.test(dataloaders=val_loader, ckpt_path='best')
        val_results_dict[f"Fold {id}"] = val_results[0]

        # Predict on test set
        predictions = trainer.predict(dataloaders=test_loader, ckpt_path='best')
        predictions = torch.cat(predictions)
        all_predictions.append(predictions.numpy())

        wandb.finish()

    output_dir = os.path.join(output_dir, group_id)
    os.makedirs(output_dir, exist_ok=True)

    # Save validation results as JSON
    with open(os.path.join(output_dir, "validation_results.json"), 'w') as json_file:
        json.dump(val_results_dict, json_file, indent=4)

    all_predictions = np.vstack(all_predictions)
    np.save(os.path.join(output_dir, f"test_predictions.npy"), all_predictions)

    print("All folds evaluated successfully.")
