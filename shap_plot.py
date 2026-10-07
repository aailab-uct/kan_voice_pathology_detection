import json

import numpy as np
from pathlib import Path

import shap
import pandas as pd
from matplotlib import pyplot as plt


def merge_all_shap_folds_binary(exp_dir: str | Path):
    """
    Merge SHAP values across all folds for binary classification.

    Returns
    -------
    shap_class0 : np.ndarray, shape (N_samples, N_features)
        Merged SHAP values for class 0.
    shap_class1 : np.ndarray, shape (N_samples, N_features)
        Merged SHAP values for class 1.
    labels : np.ndarray, shape (N_samples,)
        True labels of all test samples.
    indices : np.ndarray, shape (N_samples,)
        Original test indices from the dataset.
    """
    exp_dir = Path(exp_dir)
    shap0_list, shap1_list = [], []
    label_list, idx_list = [], []

    for fold_file in sorted(exp_dir.glob("shap_fold_*.npz")):
        data = np.load(fold_file, allow_pickle=True)

        # Determine which arrays exist
        has_class0 = "shap_values_class0" in data and data["shap_values_class0"] is not None
        has_class1 = "shap_values_class1" in data and data["shap_values_class1"] is not None
        has_single = "shap_values" in data and data["shap_values"] is not None

        # Single array — interpret as class-1 SHAPs only
        shap1 = data["shap_values"]
        print(f"shapvalues shape {data['shap_values'].shape}")
        shap0 = -shap1  # approximate opposite class effect


        labels = np.array(data["test_labels"])
        test_idx = np.array(data["test_index"])
        shap0_list.append(shap0)
        shap1_list.append(shap1)
        label_list.append(labels)
        idx_list.append(test_idx)

    if not shap0_list or not shap1_list:
        raise RuntimeError(f"No valid SHAP files found in {exp_dir}")
    shap_class0 = np.concatenate(shap0_list, axis=0)
    shap_class1 = np.concatenate(shap1_list, axis=0)
    labels_all = np.concatenate(label_list, axis=0)
    idx_all = np.concatenate(idx_list, axis=0)

    print(f"Merged {len(shap0_list)} folds → {shap_class0.shape[0]} samples, {shap_class0.shape[1]} features")

    return shap_class0, shap_class1, labels_all, idx_all


# Example usage
if __name__ == "__main__":
    exp_dir = Path("results_kan_adam_shap/g5_k3_entropy0.1_smoothing0.2_lr0.05_reg/men/20_16_2")
    #exp_dir = Path("results_kan_adam_shap/g5_k5_entropy1.0_smoothing0.1_lr0.1_reg/women/20_36_20_2")
    # --- MERGE ALL FOLDS ---
    shap_class0, shap_class1, labels, indices = merge_all_shap_folds_binary(exp_dir)

    # Save merged arrays for later reuse
    np.savez_compressed(
        exp_dir / "merged_shap_all_folds_binary.npz",
        shap_values_class0=shap_class0,
        shap_values_class1=shap_class1,
        labels=labels,
        indices=indices
    )
    print("✅ Merged SHAP (both classes) saved as merged_shap_all_folds_binary.npz")
    print(shap_class1.shape)
    features = pd.read_csv("features.csv").drop(columns=["session_id", "pathology", "sex"])
    contrast_expanded = features["spectral_contrast"].apply(lambda x: pd.Series(json.loads(x)))
    contrast_expanded = contrast_expanded.add_prefix("spectral_contrast_")
    nan = features["nan"]
    features.drop(columns=["spectral_contrast", "nan"], inplace=True)

    features = pd.concat([features, contrast_expanded], axis=1)
    print(shap_class1[:, :, 0].shape)
    print(features.head(948).shape)
    print(features.columns)
    shap.summary_plot(shap_class1[:, :, 0], features.head(948), plot_type="bar")
    print(np.mean(shap_class0[:, 0]))