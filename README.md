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

## Repository Files

### `volleyball_prediction_pipeline.py`

Main project pipeline. This script:

- Loads and merges all yearly datasets  
- Cleans and preprocesses the raw data  
- Engineers match-level predictive features  
- Trains an XGBoost classifier  
- Reports model performance metrics such as Accuracy and ROC-AUC  
- Generates feature importance plot  

### `other_models.py`

Uses the same preprocessing pipeline as the main script but trains other models:

- Logistic Regression  
- Random Forest  
- Gradient Boosting  

Outputs comparative model performance using Accuracy and ROC-AUC.

### `thesis_conf_plots.py`

Performs confidence analysis of the XGBoost model predictions. Includes:

- Confidence bucket analysis  
- Prediction certainty evaluation  
- Calibration and reliability plots  

### `generate_figures.py`

Creates visual outputs used in the final report/dissertation, such as:

- Ablation study plots  
- Comparison charts  
- Tables for LaTeX/report inclusion  

### `ablation.py`

Runs ablation experiments to test the importance of different feature groups and compares model performance under reduced feature settings.
