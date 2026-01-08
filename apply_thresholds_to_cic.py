import joblib, pandas as pd, numpy as np, os
th = joblib.load("results/thresholds.json") if False else None

import json
with open("results/thresholds.json") as f:
    t = json.load(f)
iso_thr = t.get('iso', {}).get('threshold')
# load cic mapped file
cic = pd.read_parquet("cic_features/cic_mapped.parquet") if os.path.exists("cic_features/cic_mapped.parquet") else pd.read_parquet("cic_features/cic_mapped.parquet")
# scale & score
scaler_pkg = joblib.load("models/scaler_awid.joblib")
scaler = scaler_pkg['scaler']; feats = scaler_pkg['features']
X = scaler.transform(cic[feats].fillna(0).values)
iso = joblib.load("models/iso_model.joblib")['model']
scores = -iso.decision_function(X)
cic['iso_score'] = scores
cic['iso_anom'] = (scores >= iso_thr).astype(int)
cic.to_csv("results/cic_with_iso_scores_and_labels.csv", index=False)
print("Saved labeled CIC results to results/cic_with_iso_scores_and_labels.csv")
print("Total anomalies:", int(cic['iso_anom'].sum()), "of", len(cic))
