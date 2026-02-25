# Credit Risk Prediction with Explainability & Fairness

## Overview
This project implements an end-to-end machine learning pipeline for credit risk prediction using Logistic Regression and XGBoost. It includes explainability via SHAP and fairness analysis across gender groups.

## Key Features
- Data preprocessing & feature engineering
- Logistic Regression baseline model
- XGBoost with hyperparameter tuning
- ROC-AUC evaluation
- SHAP explainability analysis
- Gender-based fairness evaluation
- Model serialization

## Results
- Logistic Regression ROC-AUC: 0.69
- XGBoost ROC-AUC: 0.67
- Identified performance gap across gender groups

## How to Run

```bash
pip install -r requirements.txt
python credit_risk_ml_project.py