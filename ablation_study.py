from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_uar_from_folder(folder: Path) -> float:
    """Load all split results from a folder and return mean max UAR."""
    uars = []

    for split_file in sorted(folder.iterdir()):
        with open(split_file, "rb") as f:
            data = pickle.load(f)
            uars.append(max(data["test_uar"]))

    return float(np.mean(uars)) if uars else np.nan


def get_folder(grid_param: int, sex: str) -> Path:
    """Return correct folder based on grid and sex."""
    base = Path("results_kan_params_5epochs")

    if sex == "women":
        return base / f"g{grid_param}_k5_entropy1.0_smoothing0.1_regedge_forward_sum" / "women" / "20_36_20_2"
    else:
        return base / f"g{grid_param}_k3_entropy0.1_smoothing0.2_regedge_forward_spline_n" / "men" / "20_16_2"


def run_ablation(grid_values):
    """Compute ablation for men and women."""
    results_women = {}
    results_men = {}

    for g in grid_values:
        # women
        folder_w = get_folder(g, "women")
        results_women[g] = load_uar_from_folder(folder_w)

        # men
        folder_m = get_folder(g, "men")
        results_men[g] = load_uar_from_folder(folder_m)

    return results_women, results_men


# === RUN ABLATION ===
g_values = [5, 6, 7, 8]
women_res, men_res = run_ablation(g_values)

print("Women:", women_res)
print("Men:", men_res)


# === PLOT RESULTS IN SEPARATE SUBPLOTS ===
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

x = np.arange(len(g_values))

women_vals = [women_res[g] for g in g_values]
men_vals   = [men_res[g]   for g in g_values]

# --- Women subplot ---
axes[0].bar(x, women_vals, color="tab:blue")
axes[0].set_xticks(x)
axes[0].set_xticklabels(g_values)
axes[0].set_ylim(0.81, 0.875)     # <--- CUSTOM Y RANGE
axes[0].set_title("Women")
axes[0].set_xlabel("Grid Parameter g")
axes[0].set_ylabel("Mean Test UAR")
axes[0].grid(axis="y", alpha=0.3)

# --- Men subplot ---
axes[1].bar(x, men_vals, color="tab:blue")
axes[1].set_xticks(x)
axes[1].set_xticklabels(g_values)
axes[1].set_ylim(0.81, 0.875)     # <--- CUSTOM Y RANGE
axes[1].set_title("Men")
axes[1].set_xlabel("Grid Parameter g")
axes[1].grid(axis="y", alpha=0.3)

#fig.suptitle("Ablation Study: Mean Max UAR per Grid Parameter", fontsize=14)
plt.tight_layout()
plt.show()

