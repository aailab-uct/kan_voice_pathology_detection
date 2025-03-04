import pickle
from pathlib import Path
import numpy as np
import tqdm
from matplotlib import pyplot as plt

best_uar = 0
actual_uar = 10 * [0]
best_arch = ""
sex = "men"
for params_setting in tqdm.tqdm(Path("..", "results_kan_adam").iterdir()):
    try:
        for arch in params_setting.joinpath(sex).iterdir():
            for idx, split_fold in enumerate(arch.iterdir()):
                data = pickle.load(open(str(split_fold), "rb"))
                actual_uar[idx] = np.max(data["test_uar"])
                #actual_uar[idx] = data["test_uar"][0]
            if np.mean(actual_uar) >= best_uar:
                best_uar = np.mean(actual_uar)
                best_settings = params_setting.name
                best_arch = arch.name
    except FileNotFoundError:
        pass
print(f"best settings {best_settings}:best arch {best_arch} - uar: {best_uar}")

loss_path = Path("results_kan_adam", best_settings, sex, best_arch)
plt.figure(figsize=(10, 6))
for idx, file_path in enumerate(loss_path.glob('*.pickle')):
    #print(f'opening {file_path.name}')

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    loss_values = data['test_loss']
    val_loss_values = data['test_uar']
    # Plot loss function values
    plt.plot(list(range(1, 201)), val_loss_values, label=f'Loss cross-val split {idx + 1}')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Function Over Epochs')
plt.legend()
plt.grid()
plt.tight_layout()
plt.xticks([1] + list(range(10, 101, 10)))
plt.xlim([1, 200])
# plt.savefig('kan_women_train.pdf', dpi=300, format='pdf')
plt.show()
