"""This script analyzes results and finds the best parameters for KAN"""
import pickle
from pathlib import Path
import numpy as np
import tqdm
from matplotlib import pyplot as plt

best_uar = 0
actual_uar = 10 * [0]
best_arch = ""
sex = "men" # Change me
for params_setting in tqdm.tqdm(list(Path("results_kan_params_5epochs").iterdir())):
    try:
        for arch in params_setting.joinpath(sex).iterdir():
            for idx, split_fold in enumerate(arch.iterdir()):
                data = pickle.load(open(str(split_fold), "rb"))
                actual_uar[idx] = np.max(data["test_uar"])
            if np.mean(actual_uar) >= best_uar:
                best_uar = np.mean(actual_uar)
                best_settings = params_setting.name
                best_arch = arch.name
    except FileNotFoundError:
        pass
print(f"best settings {best_settings}:best arch {best_arch} - uar: {best_uar}")

loss_path = Path("results_kan_params_5epochs", best_settings, sex, best_arch)
plt.figure(figsize=(10, 6))
for idx, file_path in enumerate(list(loss_path.glob('*.pickle'))):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    loss_values = data['test_loss']
    val_loss_values = data['test_uar']
    # Plot loss function values
    plt.plot(list(range(1, 6)), val_loss_values, label=f'Loss cross-val split {idx + 1}')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Function Over Epochs')
plt.legend()
plt.grid()
plt.tight_layout()
plt.xticks([1] + list(range(10, 101, 10)))
plt.xlim([1, 5])
# plt.savefig(f'kan_{sex}_train.pdf', dpi=300, format='pdf')
plt.show()
