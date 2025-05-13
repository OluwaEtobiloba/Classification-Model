# -*- coding: utf-8 -*-
"""
Created on Mon May  5 13:08:37 2025

@author: Oluwatobiloba Alao
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load the data 
df = pd.read_csv("C:/Users/user/Desktop/DS 500/results/olist_merged_data.csv")
df.columns

#2 Filter dataset category is exactly 'furniture_decor'
df_furn = df[df['product_category_name_english'] == 'furniture_decor'].copy()

# 3. Create the binary “top-two-box” target: 1 if review_score ≥ 4, else 0
df_furn['sat_t2b'] = (df_furn['review_score'] >= 4).astype(int)

# 4. Quick check
print("Total rows:", len(df_furn))
print("Class balance:\n", df_furn['sat_t2b'].value_counts(normalize=True))
print(df_furn[['review_score','sat_t2b']].head())

# 5. Choose your features
feature_cols = ['price', 'freight_value','payment_value', 'payment_installments',
                'product_photos_qty']
X = df_furn[feature_cols]
y = df_furn['sat_t2b']

# 6. Build a preprocessing+model pipeline
model_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale',   StandardScaler()),
    ('clf',     LogisticRegression(class_weight='balanced', max_iter=1000))
])

# 7. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 8. Tune the regularization strength via 5-fold CV
param_grid = {'clf__C': [0.01, 0.1, 1, 10]}
grid = GridSearchCV(model_pipe, param_grid, cv=5, scoring='roc_auc')
grid.fit(X_train, y_train)

# 9. Evaluate on the hold-out set
y_pred  = grid.predict(X_test)
y_proba = grid.predict_proba(X_test)[:, 1]

print("\nBest C:", grid.best_params_['clf__C'])
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
