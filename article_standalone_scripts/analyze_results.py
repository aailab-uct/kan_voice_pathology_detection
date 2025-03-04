"""
Script to analyze results of kan_arch_mc.py.
"""
import pickle
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    # path to results to analyze. It is expected that directory contains multiple
    # folders with different dataset and each folder contains subfolder
    # representing various architectures (kan_arch_mc.py).
    best_men = {"mcc": 0.0,
                "sensitivity": 0.0,
                "specificity": 0.0,
                "gm": 0.0,
                "uar": 0.0,
                "bm": 0.0}

    best_women = {"mcc": 0.0,
                  "sensitivity": 0.0,
                  "specificity": 0.0,
                  "gm": 0.0,
                  "uar": 0.0,
                  "bm": 0.0}

    historical_best_men = []
    historical_best_women = []
    pickled_results_path = Path(".", "results_mlp", "training_data")

    for dataset in sorted(pickled_results_path.iterdir()):
        for arch_result in dataset.iterdir():
            # print(f"evaluating architecture {arch_result}")
            best_uar = []
            best_mcc = {"mcc": [], "sensitivity": [], "specificity": [], "gm": [], "uar": [], "bm": []}
            for result in arch_result.glob("*.pickle"):
                # read result for a single train/test split
                with open(result, "rb") as f:
                    experiment_results = pickle.load(f)
                # compute UAR
                is_nan = False
                try:
                    mcc = [(tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) for tp, tn, fp, fn in zip(experiment_results["tp"],
                                                                                                                                  experiment_results["tn"],
                                                                                                                                  experiment_results["fp"],
                                                                                                                                   experiment_results["fn"])]
                    # uar_shit = [0.5 * (tp / (tp + fn) + tn / (tn + fp)) for
                    #        tp, tn, fp, fn in zip(experiment_results["tp"],
                    #                              experiment_results["tn"],
                    #                              experiment_results["fp"],
                    #                              experiment_results["fn"])]
                    # uar = [(recall + specificity) / 2 for recall, specificity in
                    #        zip(experiment_results["recall"],
                    #            experiment_results["specificity"])]
                    # select best UAR for this train/test split
                    best_uar.append(np.max(experiment_results["uar"]))
                    best_mcc["mcc"].append(np.max(mcc))
                    best_idx = np.argmax(mcc)
                    sensitivity = experiment_results["tp"][best_idx] / (experiment_results["tp"][best_idx] + experiment_results["fn"][best_idx])
                    specificity = experiment_results["tn"][best_idx] / (experiment_results["tn"][best_idx] + experiment_results["fp"][best_idx])
                    uar = (sensitivity + specificity) / 2
                    bm = sensitivity + specificity - 1
                    gm = np.sqrt(sensitivity * specificity)
                    best_mcc["sensitivity"].append(sensitivity)
                    best_mcc["specificity"].append(specificity)
                    best_mcc["gm"].append(gm)
                    best_mcc["bm"].append(bm)
                    best_mcc["uar"].append(uar)
                except KeyError:
                    print(experiment_results.keys())
                    print(f"fuckedup {result.resolve()}")
                except ZeroDivisionError:
                    is_nan = True
            # prepare results to print
            TO_PRINT = " ".join([f"{num:.4f}" for num in best_uar])
            # print the BEST UAR for each split and MEAN of BEST UAR
            # print(f"The best UAR for each split: {TO_PRINT}")
            # print(f"Mean uar: {np.mean(best_uar)}")
            if not is_nan:
                if dataset.name == "men":
                    historical_best_men.append(np.mean(best_uar))
                    if np.mean(best_mcc["mcc"]) > best_men["mcc"]:
                        best_men["mcc"] = np.mean(best_mcc["mcc"])
                        best_men["mcc_std"] = np.std(best_mcc["mcc"])
                        best_men["sensitivity"] = np.mean(best_mcc["sensitivity"])
                        best_men["sensitivity_std"] = np.std(best_mcc["sensitivity"])
                        best_men["specificity"] = np.mean(best_mcc["specificity"])
                        best_men["specificity_std"] = np.std(best_mcc["specificity"])
                        best_men["bm"] = np.mean(best_mcc["bm"])
                        best_men["bm_std"] = np.std(best_mcc["bm"])
                        best_men["gm"] = np.mean(best_mcc["gm"])
                        best_men["gm_std"] = np.std(best_mcc["gm"])
                        best_men["uar"] = np.mean(best_mcc["uar"])
                        best_men["uar_std"] = np.std(best_mcc["uar"])
                        best_men["architecture"] = arch_result.resolve()
                else:
                    historical_best_women.append(np.mean(best_uar))
                    if np.mean(best_mcc["mcc"]) > best_women["mcc"]:
                        best_women["mcc"] = np.mean(best_mcc["mcc"])
                        best_women["mcc_std"] = np.std(best_mcc["mcc"])
                        best_women["sensitivity"] = np.mean(best_mcc["sensitivity"])
                        best_women["sensitivity_std"] = np.std(best_mcc["sensitivity"])
                        best_women["specificity"] = np.mean(best_mcc["specificity"])
                        best_women["specificity_std"] = np.std(best_mcc["specificity"])
                        best_women["bm"] = np.mean(best_mcc["bm"])
                        best_women["bm_std"] = np.std(best_mcc["bm"])
                        best_women["gm"] = np.mean(best_mcc["gm"])
                        best_women["gm_std"] = np.std(best_mcc["gm"])
                        best_women["uar"] = np.mean(best_mcc["uar"])
                        best_women["uar_std"] = np.std(best_mcc["uar"])
                        best_women["architecture"] = arch_result.resolve()
    print(f"--------------------------------{arch_result.name}-----------------------------")
    print(f"BEST WOMEN: {np.max(historical_best_women)}")
    print(f"BEST MEN: {np.max(historical_best_men)}")
    print(f"BEST WOMEN DICT: {best_women}")
    print(f"BEST MEN DICT: {best_men}")
