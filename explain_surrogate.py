#  (robust, replacement)
import os, joblib, numpy as np, pandas as pd, warnings
from config import NORMAL_PCAPS
from pathlib import Path

Path("results/figs").mkdir(parents=True, exist_ok=True)

pkg = joblib.load("models/scaler_awid.joblib")
scaler = pkg['scaler']
feats = pkg['features']

df = pd.read_parquet("features/features_clean.parquet")
print("Loaded features shape:", df.shape)

# choose train set: prefer NORMAL_PCAPS if provided
if NORMAL_PCAPS and len(NORMAL_PCAPS) > 0:
    train_df = df[df['pcap_file'].isin(NORMAL_PCAPS)].copy()
    if train_df.shape[0] == 0:
        warnings.warn("NORMAL_PCAPS is set but no rows matched in features. Falling back to whole df.")
        train_df = df.copy()
else:
    warnings.warn("NORMAL_PCAPS is empty in config.py — using whole dataset as training pool (not ideal).")
    train_df = df.copy()

n_avail = len(train_df)
n_sample = min(1000, n_avail) if n_avail>0 else min(1000, len(df))
if n_avail == 0:
    raise RuntimeError("No rows available to sample for surrogate training. Check features/features_clean.parquet")

print(f"Using {n_sample} samples for surrogate training (available={n_avail}).")
train_df = train_df.sample(n=n_sample, random_state=42)

# prepare X and surrogate labels using IsolationForest (model-based labels)
X_train = scaler.transform(train_df[feats].values)
iso = joblib.load("models/iso_model.joblib")['model']
scores = -iso.decision_function(X_train)
# label top 1% (or at least one positive) as anomalies for surrogate
pct = 99
thr = np.percentile(scores, pct)
labels = (scores >= thr).astype(int)
if labels.sum() == 0:
    # ensure at least one positive
    labels[np.argsort(scores)[-1]] = 1

print("Surrogate labels distribution:", np.bincount(labels))

# Try xgboost first, fallback to RandomForest
use_xgb = False
try:
    from xgboost import XGBClassifier
    use_xgb = True
    print("xgboost available — using XGBClassifier as surrogate.")
except Exception as e:
    print("xgboost not available or failed to import — falling back to RandomForest. Error:", e)
    from sklearn.ensemble import RandomForestClassifier
    XGBClassifier = None

# train surrogate
if use_xgb:
    clf = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=42)
else:
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)

clf.fit(X_train, labels)
print("Surrogate trained.")


try:
    import shap
    shap_available = True
    print("shap available.")
except Exception as e:
    shap_available = False
    print("shap not available:", e)

if shap_available:
    # TreeExplainer works for RF and XGB
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_train)  # for binary classifier returns list for two classes sometimes
    # shap.summary_plot handles both list and array forms
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    try:
        # If shap_vals is list (binary), pass shap_vals[1] for class 1 importance
        if isinstance(shap_vals, list):
            shap.summary_plot(shap_vals[1], pd.DataFrame(X_train, columns=feats), show=False)
        else:
            shap.summary_plot(shap_vals, pd.DataFrame(X_train, columns=feats), show=False)
        plt.savefig("results/figs/shap_summary.png", bbox_inches='tight', dpi=150)
        print("Saved results/figs/shap_summary.png")
    except Exception as e:
        print("shap plotting error:", e)
        # fallback permutation importance if plot
