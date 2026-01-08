import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, auc, f1_score
from config import FEATURE_OUTPUT, MODEL_DIR, RESULT_DIR
from utils import ensure_dirs

ensure_dirs([MODEL_DIR, RESULT_DIR])

def train_isolation_forest():
    if not os.path.exists(FEATURE_OUTPUT):
        raise FileNotFoundError("features.parquet bulunamadı. feature_extraction çalıştır.")
    df = pd.read_parquet(FEATURE_OUTPUT)
    features = ['packet_count','total_bytes','avg_pkt_len','unique_src','unique_dst','mean_interarrival','rssi_mean','rssi_std']
    X = df[features].values

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    iso = IsolationForest(n_estimators=200, contamination=0.01, n_jobs=-1, random_state=42)
    iso.fit(Xs)

    scores = -iso.decision_function(Xs)  # yüksek => daha anomali

    # kaydet
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({'model': iso, 'scaler': scaler, 'features': features}, os.path.join(MODEL_DIR, "iso_model.joblib"))
    print("Model kaydedildi:", os.path.join(MODEL_DIR, "iso_model.joblib"))

    # histogram
    plt.figure(figsize=(8,4))
    plt.hist(scores, bins=120)
    plt.title("Anomaly score histogram")
    plt.xlabel("score")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "anomaly_score_hist.png"))
    print("Histogram kaydedildi:", os.path.join(RESULT_DIR, "anomaly_score_hist.png"))

    df['score'] = scores
    top = df.sort_values('score', ascending=False).head(20)
    top.to_csv(os.path.join(RESULT_DIR, "top_anomalies.csv"), index=False)
    print("Top anomaliler kaydedildi:", os.path.join(RESULT_DIR, "top_anomalies.csv"))

    return iso, scaler, df

if __name__ == "__main__":
    train_isolation_forest()
