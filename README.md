# DTSC484-GraduationProject-VolleyballAnalytics

This project explores how machine learning can be used to predict volleyball match outcomes. This was done as a part of the graduation project (DTSC484) at FLAME University.

Dependencies required - 

Data
The dataset used for this study is from the https://github.com/JeffreyRStevens/ncaavolleyballr github repository. The project uses the team-match level data from this repository for women's division I volleyball matches from 2020-2025. These files are present in the repository as: wvb_teammatch_div1_2020.csv and so on until 2025. The data was downloaded from here.

Python files
There are 5 python files in this repository.
volleyball_prediction_pipeline.py has the code that loads all data, merges it, cleans it, then trains an XGBoost Classifier that return accuracy, ROC-AUC score and a feature importance plot.
new_fiinal.py uses the same data preprocessing setup as the above python file, but trains logistic regression, random forest and gradient boosting machine on the same data and returns the corresponding accuracies adn ROC-AUC scores.
thesis_conf_plot.py does a confidence analysis of the XGBoost model.
generate_figures.py is the script that is used to generate different plots such as weekly accuracies and calibrated probabilities.
ablation.py performs the ablation study using the same data preprocessing setup.

# DTSC484 Project - Volleyball Analytics

This project explores how machine learning can be used to predict NCAA Division I women’s volleyball match outcomes using team-match level data. The project was completed by me (Arunima Srikant) as part of the DTSC484 Graduation Project at FLAME University.

## Project Overview

The objective of this project is to predict whether a team wins or loses a volleyball match using only pre-match information. The project uses historical team-match data and applies machine learning models including XGBoost, Logistic Regression, Random Forest, and Gradient Boosting.

The repository also includes scripts for confidence analysis, feature importance, calibration plots, weekly accuracy plots, and ablation studies.

## Data Access

The dataset used in this project is from the `ncaavolleyballr` GitHub repository:

https://github.com/JeffreyRStevens/ncaavolleyballr

This project uses NCAA Women’s Division I volleyball team-match data from 2020 to 2025.

The required files are:

- `wvb_teammatch_div1_2020.csv`
- `wvb_teammatch_div1_2021.csv`
- `wvb_teammatch_div1_2022.csv`
- `wvb_teammatch_div1_2023.csv`
- `wvb_teammatch_div1_2024.csv`
- `wvb_teammatch_div1_2025.csv`

Raw datasets are not uploaded directly to this repository. To run the project, download the CSV files from the source repository and place them in the correct local data folder.

Note: The current scripts use the following local path:
SEM8/Grad Project/
