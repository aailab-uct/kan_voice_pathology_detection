"""
KAN arch search script.
"""
import os
import pickle
from pathlib import Path
import random

import torch
import numpy as np
from kan import KAN

from sklearn.model_selection import  StratifiedKFold
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
best_archs = {
    "men": {
        "k": 4,
        "grid": 6,
        "entropy": 0.01,
        "smoothing": 0.0,
        "reg": "edge_forward_spline_n",
        "arch": [21,42,26,2]
    },
    "women": {
        "k": 5,
        "grid": 7,
        "entropy": 1.0,
        "smoothing": 0.0,
        "reg": "edge_forward_spline_u",
        "arch": [21,34,26,2]
    }
}

lr_list=[0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005,  0.000001, 0.000005]

for sex, best_dict in best_archs.items():
    datadir = Path("training_data", sex)
    # load dataset
    data = np.load(datadir.joinpath("datasets.npz"))
    X=data['X']
    y=data['y']

    # get the number of features
    input_size = X.shape[1]
    # define KAN architectures
    curr_arch = best_dict["arch"]
    grid = best_dict["grid"]
    k = best_dict["k"]
    entropy = best_dict["entropy"]
    smoothing = best_dict["smoothing"]
    reg = best_dict["reg"]
    str_arch = str(curr_arch).replace(",", "_").replace(" ", "").replace("[", "").replace("]", "")
    for lrs in lr_list:
        # path where to store results
        results_path = Path(".", "results_kan_adam",
                            f"g{grid}_k{k}_entropy{entropy}_smoothing{smoothing}_lr{lrs}_reg", sex, str_arch)

        # create results directory for each dataset (done when defining results_path) and evaluated architecture

        result_dir = results_path.joinpath()
        if result_dir.exists() and len(list(result_dir.iterdir())) == 10:
            continue
        result_dir.mkdir(parents=True, exist_ok=True)

        print(f"evaluating best {sex}")

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
        for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # KMeansSMOTE resampling. if 10x fails SMOTE resampling
            X_resampled, y_resampled = CustomSMOTE(random_state=RANDOM_SEED).fit_resample(X_train, y_train)
            # MinMaxScaling
            scaler = MinMaxScaler(feature_range=(-1, 1))
            X_train_scaled = scaler.fit_transform(X_resampled).astype(np.float32)
            X_test_scaled = scaler.transform(X_test).astype(np.float32)

            # KAN dataset format, load it to device
            dataset = {
                "train_input": torch.from_numpy(X_train_scaled).type(torch_dtype).to(DEVICE),
                "train_label": torch.from_numpy(y_resampled).to(DEVICE),
                "test_input": torch.from_numpy(X_test_scaled).type(torch_dtype).to(DEVICE),
                "test_label": torch.from_numpy(y_test).to(DEVICE)
            }

            # create KAN model
            model = KAN(width=curr_arch, grid=grid, k=k, seed=RANDOM_SEED,
                        auto_save=False, save_act=True)
            # load model to device
            model.to(DEVICE)
            # train model
            print(dataset["train_input"].shape, dataset["test_input"].shape)
            results = model.fit(dataset, opt="Adam", lr=lrs, lamb=0.001, lamb_entropy=entropy, steps=200,
                                batch=-1, update_grid=False,
                                metrics=(
                                    train_acc, train_uar, test_acc, test_tn, test_tp, test_fn,
                                    test_fp, test_uar
                                ), reg_metric=reg, loss_fn=torch.nn.CrossEntropyLoss(label_smoothing=smoothing))
            print(f"final test acc: {results['test_acc'][-1]}",
                    f"mean test acc: {np.mean(results['test_acc'])}",
                    f"best test uar: {np.max(results["test_uar"])} ",
                    f"best test epoch uar: {np.argmax(results['test_uar'])}",
                    f"best train epoch uar: {np.argmax(results['train_uar'])}",
                    f"best train loss epoch uar: {np.argmin(results['train_loss'])}",
                    f"best test loss epoch: {np.argmax(results['test_loss'])}")
            print(f"uar: {results['test_uar']}")

            # dump results
            with open(result_dir.joinpath(f'kan_res_{idx+1}.pickle'), "wb") as output_file:
                pickle.dump(results, output_file)
