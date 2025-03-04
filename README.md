# Kolmogorov Arnold Network-Based Voice Pathology Detection

This repository contains the code for the paper "Kolmogorov Arnold Network-Based Voice Pathology Detection"
by Jan Vrba, Jakub Steinbach, Tomáš Jirsa, Jakub Seiner, Yuwen Zeng, Kei Ichiji, Noriyasu Homma


## Requirements
For running experiments
- prepared dataset (see below)

Used libraries and software
- Python 3.12
- see requirements.txt for all dependencies


## Dataset preparation
The dataset is not included in this repository due to the license reason, but it can be downloaded from publicly available website. Firstly download the Saarbruecken Voice Database (SVD) [available here](https://stimmdb.coli.uni-saarland.de/help_en.php4). You need to download all recordings of /a/ vowel produced at normal pitch that are encoded as wav files. Due to the limitation of SVD interface, download male recordings and female recordings separately. Then create the `svd_db` folder in the root of this project and put all recordings there.

At this step we assume following folder structure:
```
kan_voice_pathology_detection
└───article_standalone_scripts
└───misc
└───src
└───svd_db
    │   1-a_n.wav
    │   2-a_n.wav
    │   ...
    │   2610-a_n.wav
```

We provide the `svd_information.csv` file that contains the information about the SVD database (age, sex, pathologies, etc.). The file is stored in the `misc` folder and contains data scraped from the SVD website.


## Running the repository
Run the files in the following order:

1) data_preprocessing.py
2) feature_extraction.py
3) datasets_generator.py
4) ml_pipeline.py (optional)
5) mlp_pipeline.py (optional)
6) kan_params_search.py
7) params_search_results.py
8) kan_adam_training.py

## Description of files in this repository:

- **article_standalone_scripts** - miscellaneous scripts to create tables with results
- **misc** - folder with additional files
    - **data_used.sha256** - checksum of the downloaded data
    - **list_of_excluded_files.csv** - list of excluded files from the SVD database with the reason of exclusion
    - **svd_information.csv** - information about the SVD database
- **results\*** - reuslts we obtained for all experiments
- **src/custom_smote.py** - script with custom SMOTE implementation
- **data_preprocessing.py** - script that removes corrupted files and duplicities
- **dataset_generator.py** - script that generates datasets from preprocessed data
- **feature_extraction.py** - script that extracts features from preprocessed data
- **kan_adam_training.py** - train the best classifiers (one for each sex) using the Adam optimizer. You need to change best_archs manually.
- **kan_params_search.py** - do the gridsearch for KAN classifier hyperparameters
- **ml_classifier_configs.py** - store configs for ML gridsearch
- **ml_pipeline.py** -  do the gridsearch for ML classifier hyperparameters
- **mlp_pipeline.py** -  do the gridsearch for MLP classifier hyperparameters
- **params_search_results.py** - analyze results from kan_params_search.py and report the best architecture. You need to change sex manually.
- **README.md** - this file
- **requirements.txt** - list of used packages
