import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from lightning import Trainer
from timm.data import resolve_model_data_config

from classifier_module import ImageClassifier
from utils import setup_dataloaders, setup_reproducability


def main(args):
    setup_reproducability(args.seed)

    device = "mps" if torch.backends.mps.is_available() else ("gpu" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    CV_fold_path = args.CV_fold_path
    model_name = args.model_name
    output_dir = args.output_dir
    num_classes = 6
    label_map = {
        'bird': 0,
        'cat': 1,
        'deer': 2,
        'dog': 3,
        'frog': 4,
        'horse': 5
    }

    all_predictions = []
    val_results_dict = {}

    weight_dir = args.ckpt_dir
    checkpoint_files = [f for f in os.listdir(weight_dir) if f.startswith(f"best_{model_name}") and f.endswith(".ckpt")]

    for checkpoint_file in checkpoint_files:
        fold_id = checkpoint_file.split('_')[-1].split('.')[0]
        checkpoint_path = os.path.join(weight_dir, checkpoint_file)
        fold_dir = os.path.join(CV_fold_path, f"fold_{fold_id}")

        if os.path.exists(checkpoint_path) and os.path.exists(fold_dir):
            print("\n--------------> Starting Evaluation for Fold " + str(fold_id))

            model = ImageClassifier(model_name=model_name, num_classes=num_classes)
            data_config = resolve_model_data_config(model)

            _, val_loader, test_loader = setup_dataloaders(data_config=data_config,
                                                           data_dir=args.data_dir,
                                                           fold_dir=fold_dir,
                                                           seed=args.seed,
                                                           batch_size=args.batch_size,
                                                           num_workers=args.num_workers,
                                                           label_map=label_map)

            checkpoint_path = os.path.join(weight_dir, f"best_{args.model_name}_{fold_id}.ckpt")
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))['state_dict']
            model.load_state_dict(checkpoint)

            trainer = Trainer(devices=1, accelerator=device)

            # Evaluate on validation set
            val_results = trainer.test(model, dataloaders=val_loader)
            val_results_dict[f"Fold {fold_id}"] = val_results[0]

            # Predict on test set
            predictions = trainer.predict(model, dataloaders=test_loader)
            predictions = torch.cat(predictions)
            all_predictions.append(predictions.numpy())

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Save validation results as JSON
    with open(os.path.join(output_dir, "validation_results.json"), 'w') as json_file:
        json.dump(val_results_dict, json_file, indent=4)

    all_predictions = np.vstack(all_predictions)
    np.save(os.path.join(output_dir, "test_predictions.npy"), all_predictions)

    print("All folds evaluated successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate CV model checkpoints")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--CV_fold_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='CV_results')
    parser.add_argument('--model_name', type=str, default='efficientnet_b3')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    main(args)
