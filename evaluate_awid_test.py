import joblib, os, pandas as pd, numpy as np
from sklearn.metrics import precision_recall_curve, auc, f1_score, classification_report
from config import FEATURE_DIR, MODEL_DIR, NORMAL_PCAPS
df = pd.read_parquet(os.path.join(FEATURE_DIR, "features_clean.parquet"))
pkg = joblib.load(os.path.join(MODEL_DIR, "scaler_awid.joblib"))
scaler = pkg['scaler']; features = pkg['features']
import torch
# define attack windows: all pcap_files not in NORMAL_PCAPS
all_pcaps = sorted(df['pcap_file'].unique().tolist())
attack_pcaps = [p for p in all_pcaps if p not in NORMAL_PCAPS]
print("Attack pcaps:", attack_pcaps)

# build test set: all windows from attack pcaps (labels=1) + some normal windows for balance if needed
test_df = df[df['pcap_file'].isin(attack_pcaps + NORMAL_PCAPS)].reset_index(drop=True)
y_true = test_df['pcap_file'].apply(lambda x: 1 if x in attack_pcaps else 0).values
X = scaler.transform(test_df[features].values)

# load models & thresholds
import json
thr = json.load(open("results/thresholds.json"))

# IsolationForest
iso = joblib.load(os.path.join(MODEL_DIR, "iso_model.joblib"))['model']
scores_iso = -iso.decision_function(X)
pred_iso = (scores_iso >= thr['iso']['threshold']).astype(int)
print("IsolationForest report:")
print(classification_report(y_true, pred_iso))
prec, rec, _ = precision_recall_curve(y_true, scores_iso)
print("PR-AUC (iso):", auc(rec, prec))

# VAE
from vae import SimpleVAE
vae = SimpleVAE(input_dim=X.shape[1], latent_dim=8)
vae.load_state_dict(torch.load(os.path.join(MODEL_DIR, "vae_awid.pth"), map_location='cpu'))
vae.eval()
import torch
with torch.no_grad():
    recon, mu, logvar = vae(torch.from_numpy(X).float())
    re = ((recon - torch.from_numpy(X).float())**2).mean(dim=1).numpy()
pred_vae = (re >= thr['vae']['threshold']).astype(int)
print("VAE report:")
print(classification_report(y_true, pred_vae))
prec, rec, _ = precision_recall_curve(y_true, re)
print("PR-AUC (vae):", auc(rec, prec))

# save per-model summary
res = {
    'iso_pr_auc': float(auc(*precision_recall_curve(y_true, scores_iso)[:2][::-1])),
    'vae_pr_auc': float(auc(*precision_recall_curve(y_true, re)[:2][::-1]))
}
import json
with open(os.path.join("results", "awid_test_summary.json"), "w") as f:
    json.dump(res, f, indent=2)
print("Saved summary:", res)
