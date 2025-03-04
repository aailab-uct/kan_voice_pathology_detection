# TODO: is this still used?
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    sexes = ["women", "men"]
    results_path = Path(".", "results")
    classifiers = [item.name for item in results_path.glob("*/*")]
    results = {"men": [],
               "women": []}
    cls_params = {}
    for classifier in classifiers:
        for sex in sexes:
            to_process = pd.read_csv(results_path.joinpath(sex, classifier, "results.csv"))

            best = to_process[to_process.mean_test_mcc == to_process.mean_test_mcc.max()].iloc[0]
            cls_params[f"{classifier}_{sex}"] = best.params
            best["classifier"] = classifier
            best = best.drop(["params", "mean_test_accuracy"])
            results[sex].append(best)

    print(pd.DataFrame(results["men"]).reset_index(drop=True))
    print(pd.DataFrame(results["women"]).reset_index(drop=True))
    print(cls_params)

    pd.DataFrame(results["men"]).reset_index(drop=True).transpose().to_latex("ml_results_men.tex")
    pd.DataFrame(results["women"]).reset_index(drop=True).transpose().to_latex("ml_results_women.tex")
