"""
Script to generate all evaluated datasets, with various features combinations
"""
import csv
from pathlib import Path
from multiprocessing import freeze_support
import json
import random
import numpy as np


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

sexes = [0, 1]


def compose_dataset(dataset_params: dict) -> None:
    """
    Function that compose a dataset based on features specified in dataset_params dictionary.
    The dataset is dumped to unique folder. Dataclass was used back in the time.. Due to
    limited audience, I went back to the dict usage.
    :param dataset_params: dictionary that specify the features to be used
    :return: None. Dataset is dumped to a file
    """
    X = []
    y = []

    with open("features.csv", newline="", encoding="utf-8") as csv_file:
        # read feature set
        dataset = csv.DictReader(csv_file, dialect="unix")
        patient: dict
        # iterate over pations
        for patient in dataset:
            patient_features = []
            patient.pop("session_id")
            # select patients according to the specified sex
            if int(patient.pop("sex")) == dataset_params["sex"] or dataset_params["sex"] is None:
                # check specified features and add them to dataset that will be dumped
                y.append(int(patient.pop("pathology")))

                for value in patient.values():
                    # if value is list pass, if is one number print it
                    if value.startswith("["):
                        value = json.loads(value)
                        for val in value:
                            patient_features.append(float(val))
                    else:
                        patient_features.append(float(value))

                X.append(patient_features)

    match dataset_params["sex"]:
        case 0:
            subdir_name = "men"
        case 1:
            subdir_name = "women"
        case None:
            subdir_name = "both"
        case _:
            raise RuntimeError()

    X = np.array(X,np.float32)
    y = np.array(y)

    dataset_path = Path(".").joinpath("training_data", subdir_name)
    dataset_path.mkdir(parents=True,exist_ok=True)

    # dump data
    np.savez(dataset_path.joinpath("datasets.npz"),
             X=X,y=y)
    print(X.shape)

    # dump dataset config
    with dataset_path.joinpath("config.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_params, f)



# pylint: disable=too-many-locals
def main() -> None:
    """
    Main function that creates two datasets based on the features specified in the script.
    :return: None
    """
    for sex_of_interest in sexes:
        # dataset configuration
        dataset_config = {
            "sex": sex_of_interest,
            "diff_pitch": True,
            "stdev_f0": True,
            "spectral_centroid": True,
            "spectral_contrast": True,
            "spectral_flatness": True,
            "spectral_rolloff": True,
            "zero_crossing_rate": True,
            "skewness": True,
            "shannon_entropy": True,
            "nan": True,
        }

        compose_dataset(dataset_config)

# pylint: enable=too-many-locals

if __name__ == "__main__":
    freeze_support()
    main()
