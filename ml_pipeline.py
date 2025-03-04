"""
Script that performs grid search for SVM across various datasets.
Datasets are balanced via KMeansSMOTE and SMV is 10 fold cross-validated.
"""
from pathlib import Path
import warnings
import os

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.metrics import make_scorer
from imblearn.metrics import geometric_mean_score
from ml_classifier_configs import get_classifier

# setup random seed to make code reproducible
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# speedup
sklearn.set_config(
    assume_finite=False,
    skip_parameter_validation=True,
)

# specification of evaluated metrics
scoring_dict = {"mcc": make_scorer(matthews_corrcoef),
                "accuracy": make_scorer(accuracy_score),
                "recall": make_scorer(recall_score),
                "specificity": make_scorer(recall_score, pos_label=0),
                "gm": make_scorer(geometric_mean_score),
                "uar": make_scorer(balanced_accuracy_score, adjusted=False),
                "bm": make_scorer(balanced_accuracy_score, adjusted=True)}

# ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning, )
warnings.simplefilter(action="once", category=sklearn.exceptions.ConvergenceWarning)
os.environ["PYTHONWARNINGS"] = "ignore"



# pylint: disable=too-many-locals
def main(sex: str = "women",
         classifier="svm_poly"):
    """
    Main function for the classifier pipeline.

    :param sex: The sex for which the classifier is trained. Default is "women".
    :param classifier: The type of classifier to use. Default is "svm_poly".
    """
    training_data = Path(".").joinpath("training_data", sex)
    results_data = Path(".").joinpath("results", sex, classifier)
    results_data.mkdir(exist_ok=True, parents=True)

    pipeline, param_grid = get_classifier(classifier, random_seed=RANDOM_SEED)

    data = np.load(training_data.joinpath("datasets.npz"))
    X=data['X']
    y=data['y']
    cross_validation = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

    # sklearn gridsearch with cross validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=cross_validation, scoring=scoring_dict,
                                n_jobs=-1, refit=False, verbose=1)
    grid_search.fit(X, y)


    # create folder with the training_dataset name to store results
    results_data.mkdir(exist_ok=True)
    # no need to write header again and again and again,...
    if results_data.joinpath("results.csv").exists():
        header = False
    else:
        header = True

    # dump gridsearch results to a results.csv
    pd.DataFrame(grid_search.cv_results_).round(6)[
        ["params", "mean_test_accuracy", "std_test_accuracy", "mean_test_recall", "std_test_recall",
        "mean_test_specificity", "std_test_specificity", "mean_test_mcc", "std_test_mcc", "mean_test_gm",
        "std_test_gm", "mean_test_uar", "std_test_uar", "mean_test_bm", "std_test_bm",]].to_csv(
        results_data.joinpath("results.csv"),
        index=False, mode="a",
        header=header,encoding="utf8",lineterminator="\n")


if __name__ == "__main__":
    for current_classifier in ["knn", "svm_poly", "svm_rbf", "gauss_nb", "random_forest", "adaboost"]:
        for current_sex in ["women", "men"]:
            print(f"Computing {current_classifier} for {current_sex}")
            main(current_sex, current_classifier)
