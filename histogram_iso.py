import matplotlib.pyplot as plt, joblib, os, numpy as np
MODEL_DIR="models"; RESULT_DIR="results"
iso = joblib.load(os.path.join(MODEL_DIR,"iso_model.joblib"))['model']
# load features.parquet used for training
import pandas as pd
df = pd.read_parquet("features/features.parquet")
pkg = joblib.load(os.path.join(MODEL_DIR,"scaler_awid.joblib"))
scaler = pkg['scaler']; feats = pkg['features']
X = scaler.transform(df[feats].fillna(0).values)
scores = -iso.decision_function(X)
plt.figure(figsize=(6,3))
plt.hist(scores, bins=100)
plt.title("IsolationForest anomaly score histogram")
plt.tight_layout()
os.makedirs(os.path.join(RESULT_DIR,"figs"), exist_ok=True)
plt.savefig(os.path.join(RESULT_DIR,"figs","iso_score_hist.png"))
plt.close()
print("Saved histogram.")
