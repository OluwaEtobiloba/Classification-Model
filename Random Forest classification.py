# -*- coding: utf-8 -*-
"""
Created on Mon May  5 14:02:17 2025

@author: Oluwatobiloba Alao
"""

import pandas as pd
import matplotlib.pyplot as plt
# modeling & preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose       import ColumnTransformer
from sklearn.impute        import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics       import classification_report, roc_auc_score

# imbalanced‐learn tools
from imblearn.pipeline            import Pipeline as ImbPipeline
from imblearn.over_sampling       import SMOTE
from imblearn.ensemble            import BalancedRandomForestClassifier

# 1. Load the data 
df = pd.read_csv("C:/Users/user/Desktop/DS 500/results/olist_merged_data.csv")
df.columns

#2 Filter dataset category is exactly 'furniture_decor'
df_furn = df[df['product_category_name_english'] == 'furniture_decor'].copy()

# 3. Create the binary “top-two-box” target: 1 if review_score ≥ 4, else 0
df_furn['sat_t2b'] = (df_furn['review_score'] >= 4).astype(int)

print('Class distribution (normalized):')
print(df_furn['sat_t2b'].value_counts(normalize=True))

# Select features
numeric_feats = ['price', 'freight_value','payment_value', 'payment_installments',
                'product_photos_qty']
cat_feats     = ['customer_state', 'seller_state']

X = df_furn[numeric_feats + cat_feats]
y = df_furn['sat_t2b']

# 4. Preprocessing pipelines
num_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale',   StandardScaler())
])
cat_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('ohe',     OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, numeric_feats),
    ('cat', cat_pipe, cat_feats)
])

# 5. Build imbalanced‐learn pipeline with SMOTE + Balanced Random Forest
pipeline = ImbPipeline([
    ('pre',   preprocessor),
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('clf',   BalancedRandomForestClassifier(n_estimators=200,
                                             random_state=42))
])

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 7. Hyperparameter tuning
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth':    [None, 10, 20]
}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)

# 8. Evaluate on hold-out set
best = grid.best_estimator_
y_pred  = best.predict(X_test)
y_proba = best.predict_proba(X_test)[:, 1]

print("Best params:", grid.best_params_)
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
