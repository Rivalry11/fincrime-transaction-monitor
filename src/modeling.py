"""Diagnostic version - logs every single step to find the bottleneck."""
 
import time
import sys
 
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    sys.stdout.flush()
 
log("Script started")
 
import pandas as pd
log("pandas imported")
 
import numpy as np
log("numpy imported")
 
import pickle
from pathlib import Path
log("stdlib imported")
 
from sklearn.model_selection import train_test_split
log("train_test_split imported")
 
from sklearn.linear_model import LogisticRegression
log("LogisticRegression imported")
 
from sklearn.preprocessing import StandardScaler
log("StandardScaler imported")
 
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_auc_score, average_precision_score,
)
log("sklearn metrics imported")
 
from xgboost import XGBClassifier
log("xgboost imported")
 
FEATURES = [
    'amount_vs_user_avg', 'amount_zscore', 'log_amount',
    'tx_count_1h', 'tx_count_24h', 'amount_sum_24h',
    'is_foreign_country', 'is_high_risk_country',
    'is_high_risk_mcc', 'is_night_tx', 'is_cnp',
]
 
Path('models').mkdir(exist_ok=True)
log("About to read CSV...")
 
df = pd.read_csv('data/transactions_featured.csv')
log(f"CSV read. Shape: {df.shape}")
 
log("Selecting features...")
X = df[FEATURES].fillna(0)
y = df['is_fraud']
log(f"X shape: {X.shape}  |  fraud rate: {y.mean():.2%}")
 
log("About to do train_test_split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
log(f"Split done. Train: {len(X_train)}  Test: {len(X_test)}")
 
log("About to scale features...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
log("Scaling done")
 
log("About to train LogisticRegression...")
lr = LogisticRegression(class_weight='balanced', max_iter=200, random_state=42, n_jobs=-1)
lr.fit(X_train_s, y_train)
log("LR trained")
 
log("About to predict with LR...")
y_proba_lr = lr.predict_proba(X_test_s)[:, 1]
y_pred_lr = (y_proba_lr >= 0.5).astype(int)
log(f"LR PR-AUC: {average_precision_score(y_test, y_proba_lr):.4f}")
 
log("About to train XGBoost...")
spw = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    scale_pos_weight=spw, eval_metric='aucpr',
    random_state=42, n_jobs=-1, tree_method='hist',
)
xgb.fit(X_train, y_train)
log("XGBoost trained")
 
y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_proba_xgb >= 0.5).astype(int)
log(f"XGB PR-AUC: {average_precision_score(y_test, y_proba_xgb):.4f}")
 
log("Saving models...")
with open('models/xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb, f)
with open('models/lr_model.pkl', 'wb') as f:
    pickle.dump({'model': lr, 'scaler': scaler}, f)
 
# Threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba_xgb)
f1 = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
thr = thresholds[np.argmax(f1[:-1])]
with open('models/threshold.pkl', 'wb') as f:
    pickle.dump(thr, f)
 
log(f"All done. Optimal threshold: {thr:.3f}")
 
print("\n" + "="*60)
print(classification_report(y_test, y_pred_xgb, target_names=['Legit', 'Fraud'], digits=3))
print(f"LR  PR-AUC:  {average_precision_score(y_test, y_proba_lr):.4f}")
print(f"XGB PR-AUC:  {average_precision_score(y_test, y_proba_xgb):.4f}")