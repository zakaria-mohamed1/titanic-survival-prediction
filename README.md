# Titanic Survival Prediction

## Overview
This project builds a supervised machine learning classification model to predict passenger survival on the Titanic dataset. The workflow covers exploratory data analysis, data cleaning, feature preprocessing, and training a Logistic Regression model using Python.

## Dataset
- Source: Titanic dataset (CSV)
- Records: 891 passengers
- Target variable: 
  - Survived (0 = Did not survive, 1 = Survived)
- Features include:
  - Passenger class (Pclass)
  - Sex
  - Age
  - Number of siblings/spouses (SibSp)
  - Number of parents/children (Parch)
  - Fare
  - Embarked port

## Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

## Exploratory Data Analysis (EDA)
- Examined dataset structure using `info()` and `describe()`
- Visualised:
  - Fare distribution using histograms
  - Passenger age by survival status using boxplots
  - Relationship between age and fare using scatter plots
  - Survival counts by passenger class using bar charts
- Analysed correlations between numerical variables using a heatmap

## Data Cleaning & Preprocessing
- Dropped columns with more than 30% missing values (Cabin)
- Filled missing values:
  - Age filled using the median
  - Embarked filled using the mode
- Encoded categorical variables (Sex, Embarked) using Label Encoding
- Scaled the Fare feature using StandardScaler

## Model Training
- Split the dataset into training and testing sets (80% / 20%)
- Trained a Logistic Regression classifier to predict passenger survival
- Used `max_iter=1000` to ensure model convergence

## Model Evaluation
- Achieved approximately **81% accuracy** on the test dataset
- Evaluated model performance using:
  - Classification report (precision, recall, F1-score)
  - Confusion matrix visualisation
- Observed stronger performance in predicting non-survivors compared to survivors

## Results
- Logistic Regression achieved ~81% classification accuracy
- Passenger class, sex, and fare showed strong influence on survival outcomes
- The model demonstrates effective baseline performance for binary classification

## Project Structure
