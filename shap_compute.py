"""
KAN arch search script + SHAP computation.
"""
import os
import pickle
from pathlib import Path
import random

import torch
import numpy as np
from kan import KAN
import shap  # NEW

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from src.customsmote import CustomSMOTE

############## Functions used as metrics
def train_acc():
    return torch.mean((torch.argmax(model(dataset["train_input"]), dim=1) == dataset["train_label"]).float())

def train_uar():
    predictions = torch.argmax(model(dataset["train_input"]), dim=1)
    labels = dataset["train_label"]
    tn = ((predictions == 0) & (labels == 0)).sum().float()
    tp = ((predictions == 1) & (labels == 1)).sum().float()
    fn = ((predictions == 0) & (labels == 1)).sum().float()
    fp = ((predictions == 1) & (labels == 0)).sum().float()
    recall = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    return 0.5 * (recall + specificity)

def test_acc():
    return torch.mean((torch.argmax(model(dataset["test_input"]), dim=1) == dataset["test_label"]).float())

def test_tp():
    predictions = torch.argmax(model(dataset["test_input"]), dim=1)
    labels = dataset["test_label"]
    return ((predictions == 1) & (labels == 1)).sum().float()

def test_tn():
    predictions = torch.argmax(model(dataset["test_input"]), dim=1)
    labels = dataset["test_label"]
    return ((predictions == 0) & (labels == 0)).sum().float()

def test_fp():
    predictions = torch.argmax(model(dataset["test_input"]), dim=1)
    labels = dataset["test_label"]
    return ((predictions == 1) & (labels == 0)).sum().float()

def test_fn():
    predictions = torch.argmax(model(dataset["test_input"]), dim=1)
    labels = dataset["test_label"]
    return ((predictions == 0) & (labels == 1)).sum().float()

def test_uar():
    predictions = torch.argmax(model(dataset["test_input"]), dim=1)
    labels = dataset["test_label"]
    tn = ((predictions == 0) & (labels == 0)).sum().float()
    tp = ((predictions == 1) & (labels == 1)).sum().float()
    fn = ((predictions == 0) & (labels == 1)).sum().float()
    fp = ((predictions == 1) & (labels == 0)).sum().float()
    recall = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    return 0.5 * (recall + specificity)

###############
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
###############

RANDOM_SEED = 42
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# PyKAN 0.1.2: float64 default to avoid LBFGS inversion issues
torch.set_default_dtype(torch.float64)
torch_dtype = torch.get_default_dtype()

set_seed(RANDOM_SEED)

datasets = Path("", "training_data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)

best_archs = {
    #BEST MEN DICT: '20_16_2', 'folder_name': PosixPath('../results_kan_params_5epochs/g5_k3_entropy0.1_smoothing0.2_regedge_forward_spline_n')}
    #"men":    {"k": 3, "grid": 5, "entropy": 0.1, "smoothing": 0.2, "reg": "edge_forward_spline_n", "arch": [20,16,2], "lr": 0.05},
    "women":  {"k": 5, "grid": 5, "entropy": 1.0,  "smoothing": 0.1, "reg": "edge_forward_sum", "arch": [20,36,20,2], "lr": 0.1}
}

lr_list = [0.1, 0.5, 0.01, 0.05, 0.001, 0.005]

for sex, best_dict in best_archs.items():
    datadir = Path("training_data", sex)
    data = np.load(datadir.joinpath("datasets.npz"))
    X = data['X']
    y = data['y']

    curr_arch = best_dict["arch"]
    grid = best_dict["grid"]
    k = best_dict["k"]
    best_lr = best_dict["lr"]
    entropy = best_dict["entropy"]
    smoothing = best_dict["smoothing"]
    reg = best_dict["reg"]
    str_arch = str(curr_arch).replace(",", "_").replace(" ", "").replace("[", "").replace("]", "")
    for lrs in lr_list:
        results_path = Path(".", "results_kan_adam_shap", f"g{grid}_k{k}_entropy{entropy}_smoothing{smoothing}_lr{lrs}_reg", sex, str_arch)
        result_dir = results_path.joinpath()
        if result_dir.exists() and len(list(result_dir.iterdir())) == 10:
            continue
        result_dir.mkdir(parents=True, exist_ok=True)

        print(f"evaluating best {sex}")

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
        for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # KMeansSMOTE resampling
            X_resampled, y_resampled = CustomSMOTE(random_state=RANDOM_SEED).fit_resample(X_train, y_train)
            # MinMaxScaling
            scaler = MinMaxScaler(feature_range=(-1, 1))
            X_train_scaled = scaler.fit_transform(X_resampled).astype(np.float32)
            X_test_scaled = scaler.transform(X_test).astype(np.float32)

            # KAN dataset format, load to device; use float64 to match KAN default
            dataset = {
                "train_input": torch.from_numpy(X_train_scaled).type(torch_dtype).to(DEVICE),
                "train_label": torch.from_numpy(y_resampled).to(DEVICE),
                "test_input":  torch.from_numpy(X_test_scaled).type(torch_dtype).to(DEVICE),
                "test_label":  torch.from_numpy(y_test).to(DEVICE)
            }

            # create & train KAN
            model = KAN(width=curr_arch, grid=grid, k=k, seed=RANDOM_SEED, auto_save=False, save_act=True)
            model.to(DEVICE)

            print(dataset["train_input"].shape, dataset["test_input"].shape)
            results = model.fit(
                dataset, opt="Adam", lr=best_lr, lamb=0.001, lamb_entropy=entropy, steps=500, batch=-1,
                update_grid=False,
                metrics=(train_acc, train_uar, test_acc, test_tn, test_tp, test_fn, test_fp, test_uar),
                reg_metric=reg, loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=smoothing)
            )

            print(
                f"final test acc: {results['test_acc'][-1]}",
                f"mean test acc: {np.mean(results['test_acc'])}",
                f"best test uar: {np.max(results['test_uar'])} ",
                f"best test epoch uar: {np.argmax(results['test_uar'])}",
                f"best train epoch uar: {np.argmax(results['train_uar'])}",
                f"best train loss epoch: {np.argmin(results['train_loss'])}",
                f"best test loss epoch: {np.argmin(results['test_loss'])}"
            )
            print(f"uar: {results['test_uar']}")

            # ===== SHAP COMPUTATION (FIXED) =====
            model.eval()
            with torch.no_grad():
                # pick small representative background
                bg_size = min(256, dataset["train_input"].shape[0])
                background = dataset["train_input"][:bg_size]

            try:
                explainer = shap.GradientExplainer(model, background)

                # test data must require grad
                test_in = dataset["test_input"].detach().clone().requires_grad_(True)
                shap_vals = explainer.shap_values(test_in)

                # Normalize output shape(s)
                if isinstance(shap_vals, list):
                    shap_np = [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.array(v) for v in
                               shap_vals]
                else:
                    shap_np = shap_vals.detach().cpu().numpy() if isinstance(shap_vals, torch.Tensor) else np.array(
                        shap_vals)

                # GradientExplainer does NOT have expected_value.
                # You can compute a proxy baseline output:
                with torch.no_grad():
                    baseline_output = model(background).detach().cpu().numpy()
                    expected_value = np.mean(baseline_output, axis=0)

                # Compute mean |SHAP| per feature
                if isinstance(shap_np, list):
                    mean_abs = [np.mean(np.abs(v), axis=0) for v in shap_np]
                else:
                    mean_abs = np.mean(np.abs(shap_np), axis=0)

                np.savez_compressed(
                    result_dir.joinpath(f"shap_fold_{idx + 1}.npz"),
                    shap_values_class0=(shap_np[0] if isinstance(shap_np, list) else None),
                    shap_values_class1=(shap_np[1] if isinstance(shap_np, list) else None),
                    shap_values=(None if isinstance(shap_np, list) else shap_np),
                    expected_value=expected_value,
                    test_index=test_index,
                    test_labels=dataset["test_label"].detach().cpu().numpy(),
                    mean_abs_class0=(mean_abs[0] if isinstance(mean_abs, list) else None),
                    mean_abs_class1=(mean_abs[1] if isinstance(mean_abs, list) else None),
                    mean_abs=(None if isinstance(mean_abs, list) else mean_abs)
                )

                results["shap_summary"] = {
                    "mean_abs_class0": (mean_abs[0].tolist() if isinstance(mean_abs, list) else None),
                    "mean_abs_class1": (mean_abs[1].tolist() if isinstance(mean_abs, list) else None),
                    "expected_value": expected_value.tolist()
                }

            except Exception as e:
                print(f"[WARN] SHAP computation failed on fold {idx + 1}: {e}")
                results["shap_summary"] = {"error": str(e)}

            # dump results (metrics and small SHAP summary)
            with open(result_dir.joinpath(f'kan_res_{idx+1}.pickle'), "wb") as output_file:
                pickle.dump(results, output_file)
