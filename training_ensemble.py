from pathlib import Path
import os
import random

import numpy as np
import pandas as pd
import torch
from kan import KAN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


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
    tn = ((predictions == 0) & (labels == 0)).sum().float()
    tp = ((predictions == 1) & (labels == 1)).sum().float()
    fn = ((predictions == 0) & (labels == 1)).sum().float()
    fp = ((predictions == 1) & (labels == 0)).sum().float()

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
    tp = ((predictions == 1) & (labels == 1)).sum().float()
    return tp


def test_tn():
    """
    Recall for the test set. That is how the PyKAN needs the metric functions.
    """
    predictions = torch.argmax(model(dataset["test_input"]), dim=1)
    labels = dataset["test_label"]
    tn = ((predictions == 0) & (labels == 0)).sum().float()
    return tn


def test_fp():
    """
    Specificity for the test. That is how the PyKAN needs the metric functions.
    """
    predictions = torch.argmax(model(dataset["test_input"]), dim=1)
    labels = dataset["test_label"]
    fp = ((predictions == 1) & (labels == 0)).sum().float()
    return fp


def test_fn():
    """
    Recall for the test set. That is how the PyKAN needs the metric functions.
    """
    predictions = torch.argmax(model(dataset["test_input"]), dim=1)
    labels = dataset["test_label"]
    fn = ((predictions == 0) & (labels == 1)).sum().float()
    return fn


def test_uar():
    """
    UAR for the test set. That is how the PyKAN needs the metric functions.
    """
    predictions = torch.argmax(model(dataset["test_input"]), dim=1)
    labels = dataset["test_label"]
    tn = ((predictions == 0) & (labels == 0)).sum().float()
    tp = ((predictions == 1) & (labels == 1)).sum().float()
    fn = ((predictions == 0) & (labels == 1)).sum().float()
    fp = ((predictions == 1) & (labels == 0)).sum().float()

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
        "predictions": predictions.detach().cpu().numpy(),
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


def prepare_data(sex):
    datadir = DATASETS_DIR / sex
    data = np.load(datadir / "datasets.npz")
    X = data["X"]
    y = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    X_train_inner, X_val, y_train_inner, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=RANDOM_SEED + 1,
        stratify=y_train,
    )

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train_inner).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    prepared_dataset = {
        "train_input": torch.from_numpy(X_train_scaled).type(torch_dtype).to(DEVICE),
        "train_label": torch.from_numpy(y_train_inner).to(DEVICE),
        "test_input": torch.from_numpy(X_val_scaled).type(torch_dtype).to(DEVICE),
        "test_label": torch.from_numpy(y_val).to(DEVICE),
    }
    val_input = prepared_dataset["test_input"]
    val_label = prepared_dataset["test_label"]
    test_input = torch.from_numpy(X_test_scaled).type(torch_dtype).to(DEVICE)
    test_label = torch.from_numpy(y_test).to(DEVICE)

    return (
        prepared_dataset,
        val_input,
        val_label,
        test_input,
        test_label,
        compute_class_weights(y_train_inner),
    )


def parse_row(row):
    arch = list(map(int, row["arch"].split("_")))
    hypers = row["hypers"].split("_")
    g = int(hypers[0].replace("g", ""))
    k = int(hypers[1].replace("k", ""))
    entropy = float(hypers[2].replace("entropy", ""))
    smoothing = float(hypers[3].replace("smoothing", ""))
    hypers[4] = hypers[4].replace("reg", "")
    regularization = "_".join(hypers[4:])
    return arch, g, k, entropy, smoothing, regularization


def build_model(arch, g, k):
    return KAN(
        width=arch,
        grid=g,
        k=k,
        seed=RANDOM_SEED,
        auto_save=False,
        save_act=True,
        device=DEVICE,
    )


def clone_state_dict(model_to_clone):
    return {
        name: tensor.detach().clone()
        for name, tensor in model_to_clone.state_dict().items()
    }


def train_single_model(row, learning_rate, class_weights, training_dataset):
    global dataset
    global model

    arch, g, k, entropy, smoothing, regularization = parse_row(row)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=smoothing,
    )

    dataset = training_dataset
    model = build_model(arch, g, k)
    best_val_loss = float("inf")
    best_model_state = None
    best_step = -1

    for step_idx in range(TRAINING_STEPS):
        step_results = model.fit(
            dataset,
            opt="Adam",
            lr=learning_rate,
            lamb=0.001,
            lamb_entropy=entropy,
            steps=1,
            batch=-1,
            update_grid=True,
            metrics=(train_uar, test_tn, test_tp, test_fn, test_fp, test_uar),
            loss_fn=loss_fn,
            reg_metric=regularization,
        )

        current_val_loss = step_results["test_loss"][-1]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_step = step_idx + 1
            best_model_state = clone_state_dict(model)
        if current_val_loss > 50 and step_idx > 10:
            break
        if step_idx - best_step > 50:
            break

    if best_model_state is None:
        return model

    best_model = build_model(arch, g, k)
    best_model.load_state_dict(best_model_state)
    return best_model


def train_models_for_learning_rate(sex, learning_rate, results_table, max_models):
    training_dataset, val_input, val_label, test_input, test_label, class_weights = prepare_data(sex)
    trained_models = []

    for model_idx, (_, row) in enumerate(results_table.head(max_models).iterrows(), start=1):
        print(f"training sex={sex}, lr={learning_rate}, model_idx={model_idx}")
        selected_model = train_single_model(
            row=row,
            learning_rate=learning_rate,
            class_weights=class_weights,
            training_dataset=training_dataset,
        )
        val_metrics = evaluate_classifier(selected_model, val_input, val_label)
        test_metrics = evaluate_classifier(selected_model, test_input, test_label)
        trained_models.append(
            {
                "row": row,
                "val_predictions": val_metrics["predictions"],
                "test_predictions": test_metrics["predictions"],
            }
        )

    return trained_models, val_label.detach().cpu().numpy(), test_label.detach().cpu().numpy()


def train_models_with_optimal_learning_rates(sex, results_table, max_models):
    trained_models_by_lr = {}
    y_val = None
    y_test = None

    for learning_rate in LEARNING_RATES:
        trained_models, y_val, y_test = train_models_for_learning_rate(
            sex=sex,
            learning_rate=learning_rate,
            results_table=results_table,
            max_models=max_models,
        )
        trained_models_by_lr[learning_rate] = trained_models

    selected_models = []
    lr_search_rows = []

    for model_idx in range(max_models):
        best_candidate = None
        for learning_rate in LEARNING_RATES:
            model_result = trained_models_by_lr[learning_rate][model_idx]
            val_metrics = evaluate_ensemble([model_result["val_predictions"]], y_val)
            lr_search_rows.append(
                {
                    "SEX": sex,
                    "MODEL_IDX": model_idx + 1,
                    "LR": learning_rate,
                    "VAL_UAR": val_metrics["uar"],
                }
            )
            if best_candidate is None or val_metrics["uar"] > best_candidate["VAL_UAR"]:
                best_candidate = {
                    "row": model_result["row"],
                    "best_lr": learning_rate,
                    "VAL_UAR": val_metrics["uar"],
                    "val_predictions": model_result["val_predictions"],
                    "test_predictions": model_result["test_predictions"],
                }

        print(
            f"best lr for sex={sex}, model_idx={model_idx + 1}: "
            f"lr={best_candidate['best_lr']}, val_uar={best_candidate['VAL_UAR']:.4f}"
        )
        selected_models.append(best_candidate)

    return selected_models, y_val, y_test, lr_search_rows


def run_search_for_sex(sex, results_table):
    sorted_table = results_table.sort_values("uar_avg", ascending=False).reset_index(drop=True)
    if len(sorted_table) < MIN_ENSEMBLE_MODELS:
        raise ValueError(f"Not enough candidate models for {sex}: found {len(sorted_table)} rows.")

    max_models = min(MAX_ENSEMBLE_MODELS, len(sorted_table))
    search_rows = []
    lr_search_rows = []
    best_setting = None

    selected_models, y_val, y_test, lr_rows = train_models_with_optimal_learning_rates(
        sex=sex,
        results_table=sorted_table,
        max_models=max_models,
    )
    lr_search_rows.extend(lr_rows)
    val_predictions_prefix = []

    for number_of_models, model_result in enumerate(selected_models, start=1):
        val_predictions_prefix.append(model_result["val_predictions"])
        if number_of_models < MIN_ENSEMBLE_MODELS:
            continue

        ensemble_metrics = evaluate_ensemble(val_predictions_prefix, y_val)
        selected_lrs = [item["best_lr"] for item in selected_models[:number_of_models]]
        search_rows.append(
            {
                "SEX": sex,
                "NUMBER_OF_MODELS": number_of_models,
                "UAR": ensemble_metrics["uar"],
                "SELECTED_LRS": ",".join(map(str, selected_lrs)),
            }
        )
        print(
            f"validation ensemble result sex={sex}, number_of_models={number_of_models}, "
            f"uar={ensemble_metrics['uar']:.4f}"
        )

        if best_setting is None or ensemble_metrics["uar"] > best_setting["VAL_UAR"]:
            best_setting = {
                "SEX": sex,
                "NUMBER_OF_MODELS": number_of_models,
                "VAL_UAR": ensemble_metrics["uar"],
                "TEST_PREDICTIONS": [item["test_predictions"] for item in selected_models[:number_of_models]],
                "TEST_LABELS": y_test,
                "SELECTED_LRS": selected_lrs,
            }

    best_test_metrics = evaluate_ensemble(
        best_setting["TEST_PREDICTIONS"],
        best_setting["TEST_LABELS"],
    )

    best_test_row = {
        "SEX": best_setting["SEX"],
        "NUMBER_OF_MODELS": best_setting["NUMBER_OF_MODELS"],
        "VAL_UAR": best_setting["VAL_UAR"],
        "SELECTED_LRS": ",".join(map(str, best_setting["SELECTED_LRS"])),
        "TEST_UAR": best_test_metrics["uar"],
        "TEST_ACCURACY": best_test_metrics["accuracy"],
        "TEST_RECALL": best_test_metrics["recall"],
        "TEST_SPECIFICITY": best_test_metrics["specificity"],
    }
    print(
        f"best test setting sex={sex}, number_of_models={best_test_row['NUMBER_OF_MODELS']}, "
        f"val_uar={best_test_row['VAL_UAR']:.4f}, test_uar={best_test_row['TEST_UAR']:.4f}"
    )

    return search_rows, best_test_row, lr_search_rows


RANDOM_SEED = 0  # You can choose any number you prefer
LEARNING_RATES = [0.01, 0.005, 0.001, 0.0005, 0.0001]
MIN_ENSEMBLE_MODELS = 3
MAX_ENSEMBLE_MODELS = 31
TRAINING_STEPS = 5000

# Set the CUBLAS_WORKSPACE_CONFIG environment variable
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# since PyKAN 0.1.2 it is necessary to magically set torch default type to float64
# to avoid issues with matrix inversion during training with the LBFGS optimizer
torch.set_default_dtype(torch.float64)
torch_dtype = torch.get_default_dtype()

set_seed(RANDOM_SEED)

# path to training datasets
DATASETS_DIR = Path("training_data")
# select computational device -> changed to CPU as it is faster for small datasets (as SVD)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)

results_by_sex = {
    "men": pd.read_csv("holdout_results_men.csv"),
    "women": pd.read_csv("holdout_results_women.csv"),
}

all_search_rows = []
best_test_rows = []
lr_search_rows = []
for sex, results_table in results_by_sex.items():
    sex_search_rows, sex_best_test_row, sex_lr_search_rows = run_search_for_sex(sex, results_table)
    all_search_rows.extend(sex_search_rows)
    best_test_rows.append(sex_best_test_row)
    lr_search_rows.extend(sex_lr_search_rows)

pd.DataFrame(all_search_rows).to_csv("ensemble_search_results_nest.csv", index=False)
pd.DataFrame(best_test_rows).to_csv("ensemble_best_test_results_nest.csv", index=False)
pd.DataFrame(lr_search_rows).to_csv("ensemble_model_lr_search_results_nest.csv", index=False)
