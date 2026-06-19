import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('data/Churn_Modelling.csv')
print('Shape:', df.shape)
print('Churn rate:', round(df['Exited'].mean() *100, 2), '%')
print(df.head(3))

df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

le=LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])
df['Geography']=le.fit_transform(df['Geography'])

X=df.drop('Exited', axis=1)
y=df['Exited']
X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
print('Train size:', len(X_train))
print('Test size: ', len(X_test))

models={
    'Logistic Regression': LogisticRegression(
        max_iter=1000, class_weight='balanced'),

    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=10,
        class_weight='balanced', random_state=42),

    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=5,
        learning_rate=0.1, random_state=42),
}

scores={}
for name, model in models.items():
    print(f'Training {name}...')
    model.fit(X_train, y_train)
    preds= model.predict(X_test)
    acc= accuracy_score(y_test, preds)
    roc=roc_auc_score(y_test, preds)
    scores[name]={'accuracy': acc, 'roc_auc': roc}
    print(f'{name} —Accuracy: {acc:.4f} | ROC-AUC: {roc:.4f}')
    print(classification_report(y_test, preds,
          target_names=['Stayed', 'Churned'], zero_division=0))
    print('-' * 60)
best_name=max(scores, key=lambda x: scores[x]['roc_auc'])
best_model=models[best_name]
print(f'Best model: {best_name}')

os.makedirs('outputs', exist_ok=True)
joblib.dump(best_model, 'outputs/churn_model.pkl')
joblib.dump(scaler, 'outputs/scaler.pkl')
final_preds=best_model.predict(X_test)
results_df=pd.DataFrame({
    'actual_churn': y_test.values,
    'predicted_churn': final_preds
})
results_df.to_csv('outputs/predictions.csv', index=False)
print('Saved: outputs/churn_model.pkl')
print('Saved: outputs/scaler.pkl')
print('Saved: outputs/predictions.csv')

fig, axes=plt.subplots(1, 3, figsize=(18, 5))
#Plot 1
counts=y.value_counts()
axes[0].bar(['Stayed', 'Churned'], counts.values,
            color=['#2455A4', '#C0392B'])
axes[0].set_title('Churn Distribution', fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')
#Plot 2
df_plot=df.copy()
df_plot['Churn'] = df_plot['Exited'].map({0: 'Stayed', 1: 'Churned'})
axes[1].hist(df_plot[df_plot['Exited']==0]['Age'], bins=30,
             alpha=0.6, label='Stayed',  color='#2455A4')
axes[1].hist(df_plot[df_plot['Exited']==1]['Age'], bins=30,
             alpha=0.6, label='Churned', color='#C0392B')
axes[1].set_title('Age Distribution by Churn', fontweight='bold')
axes[1].set_xlabel('Age')
axes[1].legend()
#Plot 3
model_names=list(scores.keys())
roc_scores=[scores[m]['roc_auc'] for m in model_names]
bars=axes[2].bar(model_names, roc_scores,
                   color=['#2455A4', '#27AE60', '#E67E22'])
axes[2].set_title('Model ROC-AUC Comparison', fontweight='bold')
axes[2].set_ylabel('ROC-AUC Score')
axes[2].set_ylim(0.5, 1.0)
for i, v in enumerate(roc_scores):
    axes[2].text(i, v + 0.005, f'{v:.3f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/plots.png', dpi=150)
plt.close()
print('Plot saved to outputs/plots.png')

def predict_churn(credit_score, geography, gender, age, tenure,
                  balance, num_products, has_cr_card,
                  is_active, estimated_salary):
    data = np.array([[credit_score, geography, gender, age, tenure,
                      balance, num_products, has_cr_card,
                      is_active, estimated_salary]])
    data_scaled = scaler.transform(data)
    pred = best_model.predict(data_scaled)[0]
    return 'WILL CHURN' if pred == 1 else 'WILL STAY'
if __name__ == '__main__':
    print(predict_churn(600, 1, 0, 45, 3,
                        120000, 1, 1, 0, 80000))
    print(predict_churn(750, 0, 1, 28, 7,
                        50000, 3, 1, 1, 95000))
