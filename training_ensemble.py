from pathlib import Path
import copy
import pandas as pd
import os
import random

import torch
import numpy as np
from kan import KAN

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.customsmote import CustomSMOTE

############## Functions used as metrics
def train_acc():
    """
    Train accuracy. That is how the PyKAN needs the metric functions.
    """
    return torch.mean((torch.argmax(model(dataset["train_input"]),
                                    dim=1) == dataset["train_label"]).float())

def train_uar():
    """
    Train UAR. That is how the PyKAN needs the metric functions.
    """
    predictions = torch.argmax(model(dataset["train_input"]), dim=1)
    labels = dataset["train_label"]
    # Calculate TP, TN, FP, FN
    tn = ((predictions == 0) & (labels == 0)).sum().float()
    tp = ((predictions == 1) & (labels == 1)).sum().float()
    fn = ((predictions == 0) & (labels == 1)).sum().float()
    fp = ((predictions == 1) & (labels == 0)).sum().float()

    # Calculate recall
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    uar = 0.5 * (recall + specificity)
    return uar

def test_acc():
    """
    Test accuracy. That is how the PyKAN needs the metric functions.
    """
    return torch.mean((torch.argmax(model(dataset["test_input"]),
                                    dim=1) == dataset["test_label"]).float())

def test_tp():
    """
    Recall for the test set. That is how the PyKAN needs the metric functions.
    """
    predictions = torch.argmax(model(dataset["test_input"]), dim=1)
    labels = dataset["test_label"]
    # Calculate TP, TN, FP, FN
    tp = ((predictions == 1) & (labels == 1)).sum().float()
    return tp

def test_tn():
    """
    Recall for the test set. That is how the PyKAN needs the metric functions.
    """
    predictions = torch.argmax(model(dataset["test_input"]), dim=1)
    labels = dataset["test_label"]
    # Calculate TP, TN, FP, FN
    tn = ((predictions == 0) & (labels == 0)).sum().float()
    return tn

def test_fp():
    """
    Specificity for the test. That is how the PyKAN needs the metric functions.
    """
    predictions = torch.argmax(model(dataset["test_input"]), dim=1)
    labels = dataset["test_label"]
    # Calculate TP, TN, FP, FN

    fp = ((predictions == 1) & (labels == 0)).sum().float()

    return fp

def test_fn():
    """
    Recall for the test set. That is how the PyKAN needs the metric functions.
    """
    predictions = torch.argmax(model(dataset["test_input"]), dim=1)
    labels = dataset["test_label"]
    # Calculate TP, TN, FP, FN
    fn = ((predictions == 0) & (labels == 1)).sum().float()

    # Calculate recall
    return fn

def test_uar():
    """
    UAR for the test set. That is how the PyKAN needs the metric functions.
    """
    predictions = torch.argmax(model(dataset["test_input"]), dim=1)
    labels = dataset["test_label"]
    # Calculate TP, TN, FP, FN
    tn = ((predictions == 0) & (labels == 0)).sum().float()
    tp = ((predictions == 1) & (labels == 1)).sum().float()
    fn = ((predictions == 0) & (labels == 1)).sum().float()
    fp = ((predictions == 1) & (labels == 0)).sum().float()

    # Calculate recall
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    uar = 0.5 * (recall + specificity)
    return uar

###############
def set_seed(seed):
    """
    Function to set seed for reproducibility.
    :param seed:
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
###############


def evaluate_classifier(model_to_eval, features, labels):
    """
    Evaluate a trained classifier on the provided features and labels.
    """
    with torch.no_grad():
        logits = model_to_eval(features)
        predictions = torch.argmax(logits, dim=1)

        tp = ((predictions == 1) & (labels == 1)).sum().item()
        tn = ((predictions == 0) & (labels == 0)).sum().item()
        fp = ((predictions == 1) & (labels == 0)).sum().item()
        fn = ((predictions == 0) & (labels == 1)).sum().item()

        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        accuracy = (tp + tn) / len(labels)

    return {
        "accuracy": accuracy,
        "uar": 0.5 * (recall + specificity),
        "recall": recall,
        "specificity": specificity,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "predictions": predictions.detach().cpu().numpy()
    }


def evaluate_ensemble(classifier_predictions, labels):
    """
    Majority-vote ensemble evaluation.
    """
    stacked_predictions = np.stack(classifier_predictions, axis=0)
    ensemble_predictions = (stacked_predictions.mean(axis=0) >= 0.5).astype(np.int64)

    tp = int(((ensemble_predictions == 1) & (labels == 1)).sum())
    tn = int(((ensemble_predictions == 0) & (labels == 0)).sum())
    fp = int(((ensemble_predictions == 1) & (labels == 0)).sum())
    fn = int(((ensemble_predictions == 0) & (labels == 1)).sum())

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / len(labels)

    return {
        "accuracy": accuracy,
        "uar": 0.5 * (recall + specificity),
        "recall": recall,
        "specificity": specificity,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compute_class_weights(labels):
    """
    Compute inverse-frequency weights from the original training labels.
    """
    class_counts = np.bincount(labels.astype(np.int64), minlength=2)
    total_samples = class_counts.sum()
    weights = total_samples / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch_dtype, device=DEVICE)

RANDOM_SEED = 42 # You can choose any number you prefe

# Set the CUBLAS_WORKSPACE_CONFIG environment variable
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# since PyKAN 0.1.2 it is necessary to magically set torch default type to float64
# to avoid issues with matrix inversion during training with the LBFGS optimizer
torch.set_default_dtype(torch.float64)
torch_dtype = torch.get_default_dtype()

set_seed(RANDOM_SEED)

# path to training datasets
datasets = Path("", "training_data")
# select computational device -> changed to CPU as it is faster for small datasets (as SVD)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)


men_results = Path("holdout_results_men.csv")
women_results = Path("holdout_results_women.csv")

men_table = pd.read_csv(men_results)
women_table = pd.read_csv(women_results)
best_arch_table = women_table.sort_values("uar_avg", ascending=False).head(5)

datadir = Path("training_data", "women")
data = np.load(datadir.joinpath("datasets.npz"))
X = data['X']
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y)
X_train_inner, X_val, y_train_inner, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=RANDOM_SEED+1, stratify=y_train)

# KMeansSMOTE resampling. if 10x fails SMOTE resampling
X_resampled, y_resampled = CustomSMOTE(random_state=RANDOM_SEED).fit_resample(
    X_train_inner, y_train_inner
)
class_weights = compute_class_weights(y_train_inner)
# MinMaxScaling
scaler = MinMaxScaler(feature_range=(-1, 1))

X_train_scaled = scaler.fit_transform(X_resampled).astype(np.float32)
X_val_scaled = scaler.transform(X_val).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

dataset = {
            "train_input": torch.from_numpy(X_train_scaled).type(torch_dtype).to(DEVICE),
            "train_label": torch.from_numpy(y_resampled).to(DEVICE),
            "test_input": torch.from_numpy(X_val_scaled).type(torch_dtype).to(DEVICE),
            "test_label": torch.from_numpy(y_val).to(DEVICE)
        }

test_input = torch.from_numpy(X_test_scaled).type(torch_dtype).to(DEVICE)
test_label = torch.from_numpy(y_test).to(DEVICE)
classifier_test_metrics = []
classifier_predictions = []
training_steps = 10000

for model_idx, (_, row) in enumerate(best_arch_table.iterrows(), start=1):
    #print(f"Arch: {row["arch"]}, hypers: {row["hypers"]}")
    arch = list(map(int,row["arch"].split("_")))
    hypers = row["hypers"].split("_")
    g = int(hypers[0].replace("g", ""))
    k = int(hypers[1].replace("k", ""))
    entropy = float(hypers[2].replace("entropy", ""))
    smoothing = float(hypers[3].replace("smoothing", ""))
    hypers[4] = hypers[4].replace("reg", "")
    regularization = "_".join(hypers[4:])
    print(f"g={g}, k={k}, entropy={entropy}, smoothing={smoothing}, regularization={regularization}, arch={arch}")
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=smoothing
    )

    model = KAN(width=arch, grid=g, k=k, seed=RANDOM_SEED,
                auto_save=False, save_act=True, device=DEVICE)
    results = {}
    best_val_loss = float("inf")
    best_model = None
    best_step = -1

    for step_idx in range(training_steps):
        step_results = model.fit(
            dataset,
            opt="LBFGS",
            lr=0.0001, #muži 0.001
            lamb=0.001,
            lamb_entropy=entropy,
            steps=1,
            batch=-1,
            update_grid=True,
            metrics=(
                train_uar, test_tn,
                test_tp, test_fn, test_fp, test_uar
            ),
            loss_fn=loss_fn,
            reg_metric=regularization
        )

        for metric_name, metric_values in step_results.items():
            results.setdefault(metric_name, []).extend(metric_values)

        current_val_loss = step_results["test_loss"][-1]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_step = step_idx + 1
            best_model = model #copy.deepcopy(model)
        if current_val_loss > 10:
            break

        if step_idx - best_step > 50 :
            break

    selected_model = best_model if best_model is not None else model
    pd.DataFrame(results).to_csv(f"pokus_kan_res_{model_idx}.csv", index=False)

    test_metrics = evaluate_classifier(selected_model, test_input, test_label)
    classifier_predictions.append(test_metrics.pop("predictions"))
    classifier_test_metrics.append({
        "model_idx": model_idx,
        "arch": row["arch"],
        "hypers": row["hypers"],
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        **test_metrics
    })
    print(
        f"best validation checkpoint model {model_idx}: "
        f"step={best_step}, val_loss={best_val_loss:.6f}"
    )
    print(
        f"test metrics model {model_idx}: "
        f"acc={test_metrics['accuracy']:.4f}, "
        f"uar={test_metrics['uar']:.4f}, "
        f"recall={test_metrics['recall']:.4f}, "
        f"specificity={test_metrics['specificity']:.4f}"
    )

ensemble_metrics = evaluate_ensemble(classifier_predictions, y_test)
pd.DataFrame(classifier_test_metrics).to_csv("ensemble_test_metrics.csv", index=False)
pd.DataFrame([ensemble_metrics]).to_csv("ensemble_majority_vote_metrics.csv", index=False)
print(
    "ensemble test metrics: "
    f"acc={ensemble_metrics['accuracy']:.4f}, "
    f"uar={ensemble_metrics['uar']:.4f}, "
    f"recall={ensemble_metrics['recall']:.4f}, "
    f"specificity={ensemble_metrics['specificity']:.4f}"
)
