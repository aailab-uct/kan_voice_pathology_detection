# Study on Effectiveness of Kolmogorov-Arnold Networks: Voice Pathology Detection



[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14967743.svg)](https://doi.org/10.5281/zenodo.14967743)



This repository contains the code for the paper "Study on Effectiveness of Kolmogorov-Arnold Networks: Voice Pathology Detection"
by Tomáš Jirsa, Jakub Steinbach, Jakub Seiner, Yuwen Zeng, Kei Ichiji, Noriyasu Homma, and Jan Vrba.


## Requirements
For running experiments
- prepared dataset (see below)

Used libraries and software
- Python 3.12
- see requirements.txt for all dependencies


## Dataset preparation
The dataset is not included in this repository due to the license reason, but it can be downloaded from publicly available website. Firstly download the Saarbruecken Voice Database (SVD) [available here](https://stimmdb.coli.uni-saarland.de/help_en.php4). You need to download all recordings of /a/ vowel produced at normal pitch that are encoded as wav files. Then create the `svd_db` folder in the root of this project and put all recordings there.

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

