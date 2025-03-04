"""
Script to analyze results of kan_pipeline.py.
"""
import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm


np.seterr(all="raise")


def main():
    """
    Main function to analyze results of KAN architecture search.
    :return: dict dictionary of best results, containing two dict for each sex
    """
    best_results = {
        "men": {
            "mcc": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "gm": 0.0,
            "uar": 0.0,
            "bm": 0.0
        },
        "women": {
            "mcc": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "gm": 0.0,
            "uar": 0.0,
            "bm": 0.0
        }
    }
    optimistic_uar = {
        "women": 0.0,
        "men": 0.0,
    }

    pickled_results_path = Path("..", "results_kan_adam")
    for kan_settings in tqdm(sorted(pickled_results_path.iterdir())):
        for dataset in sorted(kan_settings.iterdir()):
            sex = dataset.name
            assert sex in best_results
            for arch in dataset.iterdir():
                result_all_splits = {}
                for result in arch.glob("*.pickle"):
                    # read result for a single train/test split
                    # and save it into dictionary
                    idx = int(result.stem.split("_")[-1])
                    with open(result, "rb") as f:
                        split_result = pickle.load(f)
                    for key, val in split_result.items():
                        if not key in result_all_splits.keys():
                            result_all_splits[key] = [val]
                        else:
                            result_all_splits[key].append(val)

                for key, val in result_all_splits.items():
                    result_all_splits[key] = np.array(val)
                optimistic_uar[sex] = np.max([optimistic_uar[sex], np.mean(np.max(result_all_splits["test_uar"],axis=1))])
                best_idx = np.argmax(result_all_splits["test_uar"],axis=1)
                tps = result_all_splits["test_tp"][np.arange(len(best_idx)), best_idx]
                fps = result_all_splits["test_fp"][np.arange(len(best_idx)), best_idx]
                tns = result_all_splits["test_tn"][np.arange(len(best_idx)), best_idx]
                fns = result_all_splits["test_fn"][np.arange(len(best_idx)), best_idx]

                sensitivity = tps/(tps+fns)
                specificity = tns/(tns+fps)
                try:
                    mcc = (tps * tns - fps * fns) / np.sqrt((tps + fps) * (tps + fns) * (tns + fps) * (tns + fns))
                except FloatingPointError:
                    # Zero division
                    mcc = -1*np.ones_like(specificity)


                if np.mean(sensitivity/2+specificity/2) > best_results[sex]["uar"]:
                    best_results[sex]["mcc"] = np.mean(mcc)
                    best_results[sex]["mcc_std"] = np.std(mcc)
                    best_results[sex]["sensitivity"] = np.mean(sensitivity)
                    best_results[sex]["sensitivity_std"] = np.std(sensitivity)
                    best_results[sex]["specificity"] = np.mean(specificity)
                    best_results[sex]["specificity_std"] = np.std(specificity)
                    best_results[sex]["bm"] = np.mean(sensitivity + specificity - 1)
                    best_results[sex]["bm_std"] = np.std(sensitivity + specificity - 1)
                    best_results[sex]["gm"] = np.mean(np.sqrt(sensitivity * specificity))
                    best_results[sex]["gm_std"] = np.std(np.sqrt(sensitivity * specificity))
                    best_results[sex]["uar"] = np.mean(sensitivity/2+specificity/2)
                    best_results[sex]["uar_std"] = np.std(sensitivity/2+specificity/2)
                    best_results[sex]["architecture"] = arch.name
    print(f"Optimistic UAR: {optimistic_uar}")
    return best_results



if __name__ == "__main__":
    best_results_dict = main()
    print(f"BEST WOMEN: {best_results_dict["women"]["uar"]}")
    print(f"BEST MEN: {best_results_dict["men"]["uar"]}")
    print(f"BEST WOMEN DICT: {best_results_dict["women"]}")
    print(f"BEST MEN DICT: {best_results_dict["men"]}")
