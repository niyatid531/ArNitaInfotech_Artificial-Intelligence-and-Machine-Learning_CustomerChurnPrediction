# Customer Churn Prediction

Predicts whether a bank customer will churn using Machine Learning.

## Dataset
Bank Customer Churn — 10,000 customers (Kaggle)

## Models Used
- Logistic Regression — ROC-AUC: 0.7112
- Random Forest       — ROC-AUC: 0.7766 ✅ Best
- Gradient Boosting   — ROC-AUC: 0.7198

## Tools
Python, scikit-learn, pandas, matplotlib, seaborn, joblib

## Key Results
- Best Model: Random Forest
- Accuracy: 82.05%
- ROC-AUC: 0.7766
- Churn Recall: 70% (catches 70 out of every 100 churning customers)
- Top churn predictors: Age, IsActiveMember, NumOfProducts

## How to Run
1. Place Churn_Modelling.csv in data/ folder
2. pip install pandas scikit-learn matplotlib seaborn joblib
3. python classifier.py
