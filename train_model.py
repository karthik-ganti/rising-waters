"""
One-shot training script — runs the full ML pipeline and saves
floods.save (XGBoost model) and transform.save (StandardScaler).

Actual dataset columns:
  Temp, Humidity, Cloud Cover, ANNUAL, Jan-Feb, Mar-May,
  Jun-Sep, Oct-Dec, avgjune, sub, flood
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    from xgboost import XGBClassifier
    USE_XGB = True
    print('[INFO] XGBoost found.')
except ImportError:
    USE_XGB = False
    print('[INFO] XGBoost not found — using GradientBoostingClassifier.')

# ── Column names (confirmed from actual dataset) ─────────────────────────────
TARGET_COL   = 'flood'
COL_ANNUAL   = 'ANNUAL'
COL_CLOUD    = 'Cloud Cover'
COL_SEASONAL = 'Jun-Sep'
COL_TEMP     = 'Temp'
COL_HUMIDITY = 'Humidity'
FEATURE_COLS = [COL_ANNUAL, COL_CLOUD, COL_SEASONAL, COL_TEMP, COL_HUMIDITY]

# ── 1. Load dataset ───────────────────────────────────────────────────────────
print('\n[1/7] Loading dataset...')
df = pd.read_excel(os.path.join('dataset', 'flood dataset.xlsx'))
print(f'      Shape: {df.shape}  |  Columns: {df.columns.tolist()}')
print(df.head(3).to_string())

# ── 2. Handle missing values ──────────────────────────────────────────────────
print('\n[2/7] Handling missing values...')
print('      Missing before:', df.isnull().sum().sum())
for col in df.columns:
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
print('      Missing after: ', df.isnull().sum().sum())

# ── 3. Handle outliers (IQR capping) ─────────────────────────────────────────
print('\n[3/7] Capping outliers...')
for col in FEATURE_COLS:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n = ((df[col] < lo) | (df[col] > hi)).sum()
    df[col] = df[col].clip(lo, hi)
    if n:
        print(f'      {col}: {n} outlier(s) capped')
print('      Done.')

# ── 4. Split X / y ────────────────────────────────────────────────────────────
print('\n[4/7] Splitting features and target...')
X = df[FEATURE_COLS]
y = df[TARGET_COL]
print(f'      X: {X.shape}  |  y distribution: {dict(y.value_counts())}')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'      Train: {X_train.shape[0]}  Test: {X_test.shape[0]}')

# ── 5. Feature scaling ────────────────────────────────────────────────────────
print('\n[5/7] Scaling features...')
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc  = sc.transform(X_test)

# ── 6. Train & evaluate all four models ──────────────────────────────────────
print('\n[6/7] Training models...\n')

def evaluate(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    acc  = accuracy_score(y_te, pred)
    sep = '-' * (42 - len(name))
    print(f'---- {name} {sep}')
    print(f'     Accuracy : {acc * 100:.2f}%')
    print('     Confusion Matrix:')
    print(confusion_matrix(y_te, pred))
    print('     Classification Report:')
    print(classification_report(y_te, pred, target_names=['No Flood', 'Flood']))
    return model, acc

dt_model,  dt_acc  = evaluate('Decision Tree',  DecisionTreeClassifier(random_state=42),       X_train_sc, X_test_sc, y_train, y_test)
rf_model,  rf_acc  = evaluate('Random Forest',  RandomForestClassifier(n_estimators=100, random_state=42), X_train_sc, X_test_sc, y_train, y_test)
knn_model, knn_acc = evaluate('KNN (k=5)',       KNeighborsClassifier(n_neighbors=5),            X_train_sc, X_test_sc, y_train, y_test)

# scale_pos_weight corrects for class imbalance (ratio of negatives to positives)
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
spw = round(neg / pos, 2)
print(f'      XGBoost scale_pos_weight = {spw} (neg:{neg} / pos:{pos})')

if USE_XGB:
    xgb_clf = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        random_state=42, eval_metric='logloss',
        scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=1
    )
else:
    xgb_clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                          max_depth=3, random_state=42)
xgb_model, xgb_acc = evaluate('XGBoost', xgb_clf, X_train_sc, X_test_sc, y_train, y_test)

# ── Comparison summary ────────────────────────────────────────────────────────
results = {'Decision Tree': dt_acc, 'Random Forest': rf_acc, 'KNN': knn_acc, 'XGBoost': xgb_acc}
print('\n+----------------------------------------------+')
print('|       MODEL PERFORMANCE COMPARISON          |')
print('+----------------------------------------------+')
for name, acc in results.items():
    bar = '#' * int(acc * 100 / 5)
    print(f'|  {name:<22} {acc * 100:>6.2f}%  {bar}')
print('+----------------------------------------------+')
best_name = max(results, key=results.get)
best_acc  = results[best_name]
model_map = {
    'Decision Tree': dt_model,
    'Random Forest': rf_model,
    'KNN':           knn_model,
    'XGBoost':       xgb_model
}
best_model = model_map[best_name]
print(f'\nBest model: {best_name} ({best_acc * 100:.2f}%) — selected for deployment.\n')

# ── 7. Save best model and scaler ────────────────────────────────────────────
print('[7/7] Saving model and scaler...')
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, os.path.join('models', 'floods.save'))
joblib.dump(sc,         os.path.join('models', 'transform.save'))
print(f'      models/floods.save    OK  ({best_name})')
print('      models/transform.save OK')
print(f'\nFeature order for app.py: {FEATURE_COLS}')
print(f'Saved model accuracy: {best_acc * 100:.2f}%')
print('\nTraining complete. Run: python app.py')
