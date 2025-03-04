"""
This module contains a custom implementation of SMOTE algorithm. The custom implementation
is based on the KMeansSMOTE algorithm from the imbalanced-learn library with fallback to
standard SMOTE in case KMeansSMOTE fails 10 times.
"""
import warnings

from imblearn.base import BaseSampler
from imblearn.over_sampling import SMOTE, KMeansSMOTE


warnings.simplefilter(action='ignore', category=FutureWarning, )

#pylint: disable=broad-exception-caught

class CustomSMOTE(BaseSampler):
    """Class that implements KMeansSMOTE oversampling. Due to initialization of KMeans
    there are 10 tries to resample the dataset. Then standard SMOTE is applied.
    """
    _sampling_type = "over-sampling"

    def __init__(self, random_state=None, kmeans_args=None, smote_args=None):
        super().__init__()
        self.random_state = random_state
        self.kmeans_args = kmeans_args if kmeans_args is not None else {}
        self.smote_args = smote_args if smote_args is not None else {}
        if random_state is not None:
            self.kmeans_args["random_state"] = random_state
            self.smote_args["random_state"] = random_state
        self.kmeans_smote = KMeansSMOTE(**self.kmeans_args)
        self.smote = SMOTE(**self.smote_args)

    def _fit_resample(self, X, y):
        resample_try = 0
        while resample_try < 10:
            try:
                X_res, y_res = self.kmeans_smote.fit_resample(X, y)
                return X_res, y_res
            except Exception:
                # dont care about exception, KmeansSMOTE failed
                if "random_state" in self.kmeans_args:
                    self.kmeans_args["random_state"] += 1
                self.kmeans_smote = KMeansSMOTE(**self.kmeans_args)
                resample_try += 1
        X_res, y_res = self.smote.fit_resample(X, y)
        return X_res, y_res
