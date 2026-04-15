import time
print(f"[{time.strftime('%H:%M:%S')}] Starting...", flush=True)

import numpy as np
from xgboost import XGBClassifier

print(f"[{time.strftime('%H:%M:%S')}] Imports done", flush=True)

X = np.random.rand(10000, 10)
y = (np.random.rand(10000) > 0.98).astype(int)

print(f"[{time.strftime('%H:%M:%S')}] Data ready, training...", flush=True)

model = XGBClassifier(n_estimators=50, tree_method='hist', n_jobs=-1)
model.fit(X, y)

print(f"[{time.strftime('%H:%M:%S')}] Done!", flush=True)