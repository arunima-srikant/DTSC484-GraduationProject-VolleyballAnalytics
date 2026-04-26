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
