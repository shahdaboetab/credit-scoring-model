# Credit Scoring Model 💳

A machine learning project to predict an individual's creditworthiness (good/bad credit risk) using classification algorithms.

## Objective
Build and compare models to assess credit risk and help automate lending decisions.

## Dataset
- **Source**: Statlog (German Credit Data) from UCI Machine Learning Repository
- 1000 samples, 20 features (credit amount, age, job, housing, etc.)
- Binary target: Good (0) / Bad (1) credit

## Algorithms Used
- Logistic Regression (baseline)
- Decision Tree
- Random Forest (best performing)

## Key Features
- Data preprocessing and exploratory analysis
- Model training and evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
- Feature importance visualization
- Comparison of model performance

## Results
- Random Forest achieved the highest ROC-AUC: ~0.82
- Handles class imbalance effectively


