#(robust + mean_interarrival unit check + RSSI fill)
import os, joblib, numpy as np, pandas as pd, json
from pathlib import Path


MODEL_DIR = "models"
CIC_FEATURE_DIR = "cic_features"
OUT_CSV = os.path.join("results", "cic_top_anomalies_iso.csv")
OUT_SUM = os.path.join("results", "cic_cross_test_summary.json")
Path("results").mkdir(exist_ok=True)

# load cic mapped
cic_p = os.path.join(CIC_FEATURE_DIR, "cic_mapped.parquet")
if not os.path.exists(cic_p):
    raise FileNotFoundError(f"{cic_p} not found. Run map_cic_to_awid.py first.")
df = pd.read_parquet(cic_p)
print("Loaded cic_mapped.parquet shape:", df.shape)

# load scaler & AWID features
scaler_pkg = joblib.load(os.path.join(MODEL_DIR, "scaler_awid.joblib"))
scaler = scaler_pkg['scaler']
awid_features = scaler_pkg['features']
print("AWID features:", awid_features)

# determine common features
common = [f for f in awid_features if f in df.columns]
print("Common features:", common)
if len(df) == 0:
    raise RuntimeError("CIC mapped file has 0 rows. Check mapping.")
if not common:
    raise RuntimeError("No common features found between AWID and CIC mapped.")

# Quick diagnostics
print("NaN counts (common):")
print(df[common].isna().sum())

# 1) mean_interarrival unit check & normalize if needed
if 'mean_interarrival' in common:
    med = df['mean_interarrival'].median(skipna=True)
    print("mean_interarrival median:", med)
    # heuristic: if median is huge (>1e6), assume microseconds and convert to seconds
    if pd.notnull(med) and med >= 1e6:
        print("Assuming mean_interarrival is in microseconds — converting to seconds (divide by 1e6).")
        df['mean_interarrival'] = df['mean_interarrival'] / 1_000_000.0
    # if median extremely large (>=1e9) could be nanosec -> divide by 1e9 etc (not automatic)
    elif pd.notnull(med) and med >= 1e9:
        print("Warning: mean_interarrival very large (>=1e9) — check units manually.")

# 2) Fill rssi_mean/rssi_std using AWID medians if all NaN
rssi_cols = [c for c in ['rssi_mean','rssi_std'] if c in common]
if rssi_cols:
    all_nan = all(df[c].isna().all() for c in rssi_cols)
    if all_nan:
        # compute AWID medians from cleaned AWID features
        awid_feat_file = os.path.join("features", "features_clean.parquet")
        if os.path.exists(awid_feat_file):
            awid_df = pd.read_parquet(awid_feat_file)
            from config import NORMAL_PCAPS
            if NORMAL_PCAPS:
                med_source = awid_df[awid_df['pcap_file'].isin(NORMAL_PCAPS)]
                if med_source.shape[0] == 0:
                    med_source = awid_df
            else:
                med_source = awid_df
            medians = med_source[awid_features].median()
            for c in rssi_cols:
                fillval = medians.get(c, 0.0)
                print(f"Filling {c} with AWID median:", fillval)
                df[c] = df[c].fillna(fillval)
        else:
            # fallback: fill with 0
            print("AWID cleaned features not found -> filling RSSI cols with 0.")
            for c in rssi_cols:
                df[c] = df[c].fillna(0.0)

# 3) For any remaining NaNs in common features, fill with AWID medians (robust)
# compute AWID medians once
awid_feat_file = os.path.join("features", "features_clean.parquet")
if os.path.exists(awid_feat_file):
    awid_df = pd.read_parquet(awid_feat_file)
    from config import NORMAL_PCAPS
    if NORMAL_PCAPS:
        med_source = awid_df[awid_df['pcap_file'].isin(NORMAL_PCAPS)]
        if med_source.shape[0] == 0:
            med_source = awid_df
    else:
        med_source = awid_df
    awid_medians = med_source[awid_features].median()
else:
    awid_medians = pd.Series({f:0.0 for f in awid_features})

# fill remaining NaNs for common features
for c in common:
    if df[c].isna().any():
        fillv = awid_medians.get(c, 0.0)
        df[c] = df[c].fillna(fillv)

X = df[common].values
print("Prepared X shape:", X.shape)

Xs = scaler.transform(X)

iso = joblib.load(os.path.join(MODEL_DIR, "iso_model.joblib"))['model']
scores_iso = -iso.decision_function(Xs)
df['anomaly_score_iso'] = scores_iso
df_sorted = df.sort_values('anomaly_score_iso', ascending=False)

df_sorted.head(200).to_csv(OUT_CSV, index=False)
print("Saved top anomalies to:", OUT_CSV)

summary = {
    "n_cic_rows": int(len(df)),
    "n_used_features": len(common),
    "top_score_mean": float(df_sorted['anomaly_score_iso'].head(20).mean()) if len(df_sorted)>0 else None
}
with open(OUT_SUM, "w") as f:
    json.dump(summary, f, indent=2)
print("Saved summary to:", OUT_SUM)
