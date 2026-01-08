import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import FEATURE_DIR, NORMAL_PCAPS, MODEL_DIR
import os

IN = os.path.join(FEATURE_DIR, "features_clean.parquet")
OUT_SCALER = os.path.join(MODEL_DIR, "scaler_awid.joblib")

print("Loading features:", IN)
df = pd.read_parquet(IN)

if not NORMAL_PCAPS:
    raise RuntimeError("NORMAL_PCAPS in config.py is empty. Run list_available_pcaps.py and set NORMAL_PCAPS to the normal pcap filenames.")

train_df = df[df['pcap_file'].isin(NORMAL_PCAPS)]
features = ['packet_count','total_bytes','avg_pkt_len','unique_src','unique_dst','mean_interarrival','rssi_mean','rssi_std']
features = [f for f in features if f in train_df.columns]

X = train_df[features].values
scaler = StandardScaler().fit(X)
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump({'scaler':scaler, 'features':features}, OUT_SCALER)
print("Scaler saved to:", OUT_SCALER)
print("Features used:", features)
