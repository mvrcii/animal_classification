import json
import os.path
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def save_predict(predict: np.ndarray):
    assert type(predict) == np.ndarray, "Wrong type, should be 'np.ndarray'!"
    assert predict.shape == (6000,), "Wrong shape, should be '(6000,)'!"
    assert predict.dtype == np.int64, "Wrong data type, should be 'dtype('int64')'"

    print("All checks passed! Saving files.")

    with open('predictions.npy', 'wb') as f:
        np.save(f, predict)
    print('Saving completed')


# Load images from the numpy file
image_data_path = '../data/test_features.npy'
images = np.load(image_data_path)
class_labels = ['Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse']

pred_path = 'predictions.npy'
if os.path.isfile(pred_path):
    weighted_ensemble_predictions = np.load('predictions.npy')
else:
    # Load the validation results
    with open('validation_results.json') as file:
        val_results = json.load(file)

    # Extract F1-scores and calculate weights
    f1_scores = np.array([val_results[f"Fold {i + 1}"]["val_f1"] for i in range(10)])
    weights = f1_scores / np.sum(f1_scores)

    # Load predictions
    predictions_path = 'test_predictions_10fold.npy'
    all_predictions = np.load(predictions_path)
    reshaped_predictions = all_predictions.reshape(10, 6000, 6)

    # Calculate the simple average ensemble predictions
    simple_average_predictions = np.mean(reshaped_predictions, axis=0)
    simple_ensemble_predictions = np.argmax(simple_average_predictions, axis=1)

    # Calculate the weighted average of predictions
    weighted_predictions = np.tensordot(weights, reshaped_predictions, axes=[0, 0])
    weighted_ensemble_predictions = np.argmax(weighted_predictions, axis=1)

    # Count predictions for the simple ensemble
    simple_ensemble_class_counts = np.zeros(6, dtype=int)
    for j in range(6):
        simple_ensemble_class_counts[j] = np.sum(simple_ensemble_predictions == j)

    # Count predictions for the weighted ensemble
    weighted_ensemble_class_counts = np.zeros(6, dtype=int)
    for j in range(6):
        weighted_ensemble_class_counts[j] = np.sum(weighted_ensemble_predictions == j)

    # Aggregate class predictions for each fold
    class_counts_per_fold = np.zeros((10, 6), dtype=int)
    for i in range(10):
        fold_predictions = np.argmax(reshaped_predictions[i], axis=1)
        for j in range(6):
            class_counts_per_fold[i, j] = np.sum(fold_predictions == j)

    # Append both ensemble counts to the class_counts matrix
    class_counts = np.vstack([class_counts_per_fold, simple_ensemble_class_counts, weighted_ensemble_class_counts])

    # Define labels
    fold_labels = [f"Fold {i + 1}" for i in range(10)] + ['Simple Ensemble', 'Weighted Ensemble']

    # Create the heatmap
    plt.figure(figsize=(12, 9))
    sns.heatmap(class_counts, annot=True, fmt="d", cmap="Blues", cbar=True, xticklabels=class_labels,
                yticklabels=fold_labels)

    plt.title('Prediction Frequency per Class Across Folds and Ensembles')
    plt.xlabel('Class')
    plt.ylabel('Fold / Ensemble')
    plt.show()

    save_predict(weighted_ensemble_predictions)

random_indices = random.sample(range(images.shape[0]), 9)
plt.figure(figsize=(6, 6))
for i, idx in enumerate(random_indices):
    plt.subplot(3, 3, i + 1)  # Arrange subplots in 2 rows and 5 columns
    plt.imshow(images[idx].reshape(32, 32), cmap='gray')  # Display image, reshaping as necessary
    plt.title(class_labels[weighted_ensemble_predictions[idx]],
              fontdict={'fontsize': 9})
    plt.axis('off')
plt.tight_layout(pad=0.7, h_pad=0.5)
plt.show()
