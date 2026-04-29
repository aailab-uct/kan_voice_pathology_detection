from pathlib import Path
import pandas as pd
from tqdm import tqdm

results_path = Path("results_kan_params_5epochs_nested")
table_men = pd.DataFrame({"hypers":[], "arch": [], "f1": [], "f2": [], "f3": [], "f4": [], "f5": [], "uar_avg": []})
table_women = pd.DataFrame({"hypers":[], "arch": [], "f1": [], "f2": [], "f3": [], "f4": [], "f5": [], "uar_avg": []})
result = {}
for hyper in tqdm(sorted(results_path.iterdir())):

    for sex in sorted(hyper.iterdir()):
        for arch in sorted(sex.iterdir()):
            result["hypers"] = str(hyper).split("/")[-1]
            result["arch"] = arch.stem

            for fold in sorted(arch.iterdir()):
                data = pd.read_csv(fold, sep=",")
                result[f"f{fold.stem.split('_')[-1]}"] = data["test_uar"].max()

            arch_result = pd.DataFrame([result])
            arch_result["uar_avg"] = arch_result[["f1", "f2", "f3", "f4", "f5"]].mean(axis=1)
            if "women" in sex.stem:
                table_women = pd.concat([table_women, arch_result], ignore_index=True)
            else:
                table_men = pd.concat([table_men, arch_result], ignore_index=True)
table_men.to_csv("holdout_results_men.csv", index=False)
table_women.to_csv("holdout_results_women.csv", index=False)
