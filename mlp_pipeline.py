"""
This script performs architecture search and evaluation for a Multilayer Perceptron (MLP) model on multiple datasets.
It includes functions for setting seeds for reproducibility, defining the MLP model, training and evaluating the model,
and a main function to orchestrate the entire process.
"""
import pickle
from pathlib import Path
import random
import numpy as np
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

from src.customsmote import CustomSMOTE

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

RANDOM_SEED = 42 # You can choose any number you prefer
set_seed(RANDOM_SEED)

# select computational device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The {DEVICE} will be used for the computation..")
torch.set_default_dtype(torch.float32)

# Define the MLP model
class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) neural network model.
    """
    def __init__(self, layer_sizes):
        """
        Initialize the MLP model.
        :param layer_sizes: (list of int) List containing the number of neurons in each layer.
        """
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(layer_sizes[-1], 2))  # Two output neuron
        #layers.append(nn.Softmax(dim=1))
        #layers.append(nn.Linear(layer_sizes[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the MLP model.
        :param x: (torch.Tensor) The input tensor to the model.
        :returns: (torch.Tensor) The output tensor from the model.
        """
        return self.network(x)


def train_and_evaluate(model, train_loader, val_loader, epochs, device):
    """
    Train and evaluate the MLP model.
    :param model: (nn.Module) The MLP model to train.
    :param train_loader: (DataLoader) The DataLoader for the training set.
    :param val_loader: (DataLoader) The DataLoader for the validation set.
    :param epochs: (int) The number of epochs to train the model.
    :param device: (torch.device) The device to use for training.
    :returns: (dict) A dictionary containing the evaluation metrics
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS(model.parameters())
    metrics = {"tp": [],
               "tn": [],
               "fp": [],
               "fn": [],
               "uar": []}
    loss_values = []
    for _ in range(epochs):
        # Training phase
        model.train()

        def closure():
            optimizer.zero_grad()
            outputs = model(train_loader.dataset.tensors[0].to(device))
            loss = criterion(outputs.squeeze(), train_loader.dataset.tensors[1].to(device))
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        loss_values.append(loss.cpu().detach().numpy())

        # Validation phase
        model.eval()
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x.to(device)).squeeze()
                predictions = torch.argmax(outputs, dim=1)
                #predictions = torch.sigmoid(outputs) > 0.5
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        conf_matrix = confusion_matrix(val_targets, val_predictions)
        tn, fp, fn, tp = conf_matrix.ravel()
        uar = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
        metrics["tp"].append(tp)
        metrics["tn"].append(tn)
        metrics["fp"].append(fp)
        metrics["fn"].append(fn)
        metrics["uar"].append(uar)
    metrics["loss"] = loss_values
    return metrics


def main():
    """
    Main function to perform MLP architecture search and evaluation on datasets.
    """
    datasets = Path("", "training_data")
    for datadir in datasets.iterdir():
        sex = datadir.stem
        print(f"evaluating {sex}")
        # load dataset
        data = np.load(datadir.joinpath("datasets.npz"))
        X=data['X']
        y=data['y']

        # path where to store results
        results_path = Path(".", "results_mlp", sex)
        results_path.mkdir(parents=True,exist_ok=True)
        # get the number of features
        input_size = X.shape[1]
        # define MLP architectures
        steps = list(np.linspace(0, 2, 21))
        mlp_archs = []
        for first in steps:
            first_layer = input_size * 2 - int(first * input_size)
            if first_layer > 0:
                mlp_archs.append([input_size, first_layer, 2])
            for second in steps:
                second_layer = input_size * 2 - int(second * input_size)
                if first_layer >= second_layer > 0:
                    mlp_archs.append([input_size, first_layer, second_layer, 2])
        print(mlp_archs)

        for arch in mlp_archs:
            best_uar = []

            # create results directory for each dataset and evaluated architecture
            result_dir = results_path.joinpath(str(arch).replace(",", "_").replace(" ", ""))
            result_dir.mkdir(parents=True, exist_ok=True)

            print(f"evaluating {str(arch)}")

            # For CV
            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
            for idx, (train_index, test_index) in tqdm(enumerate(skf.split(X, y))):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # KMeansSMOTE resampling. if 10x fails SMOTE resampling
                X_resampled, y_resampled = CustomSMOTE(random_state=RANDOM_SEED).fit_resample(X_train, y_train)
                # MinMaxScaling
                scaler = MinMaxScaler(feature_range=(-1, 1))
                X_train_scaled = scaler.fit_transform(X_resampled)
                X_test_scaled = scaler.transform(X_test)



                # Create DataLoader for training and validation sets
                train_dataset = TensorDataset(torch.from_numpy(X_train_scaled).to(DEVICE),
                                            torch.from_numpy(y_resampled).type(torch.long).to(DEVICE))
                val_dataset = TensorDataset(torch.from_numpy(X_test_scaled).to(DEVICE),
                                            torch.from_numpy(y_test).type(torch.long).to(DEVICE))

                train_loader = DataLoader(train_dataset, batch_size=len(y_train), shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=len(y_test), shuffle=False)
                # create model
                model = MLP(arch).to(DEVICE)
                results = train_and_evaluate(model, train_loader, val_loader, epochs=50, device=DEVICE)
                best_uar.append(np.max(results["uar"]))
                with open(result_dir.joinpath(f'mlp_res_{idx+1}.pickle'), "wb") as output_file:
                    pickle.dump(results, output_file)
            print(f"mean uar: {np.mean(best_uar)}")

if __name__ == "__main__":
    main()
