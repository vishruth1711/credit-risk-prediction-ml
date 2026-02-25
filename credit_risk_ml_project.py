import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# -------------------------------
# 1. Load Dataset
# -------------------------------

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

columns = [
    "Status", "Duration", "CreditHistory", "Purpose", "CreditAmount",
    "Savings", "Employment", "InstallmentRate", "PersonalStatusSex",
    "OtherDebtors", "Residence", "Property", "Age", "OtherInstallmentPlans",
    "Housing", "ExistingCredits", "Job", "Dependents", "Telephone",
    "ForeignWorker", "Target"
]

data = pd.read_csv(url, sep=' ', names=columns)

# Convert target: 1=Good, 2=Bad → 0/1
data["Target"] = data["Target"].map({1:1, 2:0})

print("Dataset Shape:", data.shape)

# -------------------------------
# 2. Feature Engineering
# -------------------------------

# Extract gender from PersonalStatusSex
data["Gender"] = data["PersonalStatusSex"].apply(lambda x: "Male" if x in ["A91","A93","A94"] else "Female")

# Drop original column
data = data.drop(columns=["PersonalStatusSex"])

# Identify categorical and numerical features
categorical_cols = data.select_dtypes(include='object').columns.tolist()
categorical_cols.remove("Gender")

numerical_cols = data.select_dtypes(include=['int64']).columns.tolist()
numerical_cols.remove("Target")

# -------------------------------
# 3. Preprocessing Pipeline
# -------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", LabelEncoder(), [])  # We encode manually below
    ],
    remainder='passthrough'
)

# Encode categorical variables manually
for col in categorical_cols + ["Gender"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

X = data.drop("Target", axis=1)
y = data["Target"]

# -------------------------------
# 4. Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 5. Model 1: Logistic Regression
# -------------------------------

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

log_preds = log_model.predict(X_test)

print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, log_preds))
print("ROC-AUC:", roc_auc_score(y_test, log_preds))
print(classification_report(y_test, log_preds))

# -------------------------------
# 6. Model 2: XGBoost
# -------------------------------

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1]
}

grid = GridSearchCV(xgb, param_grid, cv=3, scoring='roc_auc')
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

xgb_preds = best_model.predict(X_test)

print("\nXGBoost Results")
print("Accuracy:", accuracy_score(y_test, xgb_preds))
print("ROC-AUC:", roc_auc_score(y_test, xgb_preds))
print(classification_report(y_test, xgb_preds))

# -------------------------------
# 7. SHAP Explainability
# -------------------------------

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_summary.png", bbox_inches='tight')
plt.close()

print("SHAP plot saved as shap_summary.png")

# -------------------------------
# 8. Fairness Check (Gender Bias)
# -------------------------------

X_test_copy = X_test.copy()
X_test_copy["Actual"] = y_test
X_test_copy["Predicted"] = xgb_preds

male_accuracy = accuracy_score(
    X_test_copy[X_test_copy["Gender"]==1]["Actual"],
    X_test_copy[X_test_copy["Gender"]==1]["Predicted"]
)

female_accuracy = accuracy_score(
    X_test_copy[X_test_copy["Gender"]==0]["Actual"],
    X_test_copy[X_test_copy["Gender"]==0]["Predicted"]
)

print("\nFairness Analysis:")
print("Male Accuracy:", male_accuracy)
print("Female Accuracy:", female_accuracy)

# -------------------------------
# 9. Save Model
# -------------------------------

import os

save_path = os.path.join(os.getcwd(), "credit_risk_model.pkl")
joblib.dump(best_model, save_path)

print("Model saved at:", save_path)
print("Model Saved Successfully!")