import os, joblib, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, precision_recall_fscore_support

ROOT = "."
MODEL_DIR = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")
FEATURE_DIR = os.path.join(ROOT, "features")

# load scaler & features & cleaned features DF used in evaluation mix
scaler_pkg = joblib.load(os.path.join(MODEL_DIR, "scaler_awid.joblib"))
scaler = scaler_pkg['scaler']; feats = scaler_pkg['features']

df = pd.read_parquet(os.path.join(FEATURE_DIR, "features_clean.parquet"))

from config import NORMAL_PCAPS
test_df = df[~df['pcap_file'].isin(NORMAL_PCAPS)]
normal_df = df[df['pcap_file'].isin(NORMAL_PCAPS)]
sample_norm = normal_df.sample(min(len(test_df), len(normal_df)), random_state=42)
mix_df = pd.concat([test_df, sample_norm])
y_true = mix_df['pcap_file'].apply(lambda x: 1 if x not in NORMAL_PCAPS else 0).values
X_mix = scaler.transform(mix_df[feats].fillna(0).values)

os.makedirs(os.path.join(RESULT_DIR, "figs"), exist_ok=True)

# IsolationForest
iso_pkg = joblib.load(os.path.join(MODEL_DIR, "iso_model.joblib"))
iso = iso_pkg['model']
scores_iso = -iso.decision_function(X_mix)
p_iso, r_iso, _ = precision_recall_curve(y_true, scores_iso)
auc_iso = auc(r_iso, p_iso)
plt.figure(figsize=(6,4))
plt.plot(r_iso, p_iso, label=f"Iso (PR-AUC={auc_iso:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR,"figs","pr_iso.png"))
plt.close()

# OneClassSVM
oc_pkg = joblib.load(os.path.join(MODEL_DIR, "ocsvm_model.joblib"))
oc = oc_pkg['model']
scores_oc = -oc.score_samples(X_mix)
p_oc, r_oc, _ = precision_recall_curve(y_true, scores_oc)
auc_oc = auc(r_oc, p_oc)
plt.figure(figsize=(6,4))
plt.plot(r_oc, p_oc, label=f"OC-SVM (PR-AUC={auc_oc:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR,"figs","pr_ocsvm.png"))
plt.close()

# VAE
from vae import SimpleVAE
import torch
vae = SimpleVAE(input_dim=X_mix.shape[1], latent_dim=8)
vae.load_state_dict(torch.load(os.path.join(MODEL_DIR,"vae_awid.pth"), map_location='cpu'))
vae.eval()
with torch.no_grad():
    Xt = torch.from_numpy(X_mix).float()
    recon, mu, logvar = vae(Xt)
    re = ((recon - Xt)**2).mean(dim=1).numpy()
p_vae, r_vae, _ = precision_recall_curve(y_true, re)
auc_vae = auc(r_vae, p_vae)
plt.figure(figsize=(6,4))
plt.plot(r_vae, p_vae, label=f"VAE (PR-AUC={auc_vae:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR,"figs","pr_vae.png"))
plt.close()

# Combined PR plot
plt.figure(figsize=(6,4))
plt.plot(r_iso, p_iso, label=f"Iso ({auc_iso:.3f})")
plt.plot(r_oc, p_oc, label=f"OC-SVM ({auc_oc:.3f})")
plt.plot(r_vae, p_vae, label=f"VAE ({auc_vae:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curves Comparison")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR,"figs","pr_comparison.png"))
plt.close()

print("Saved PR plots to", os.path.join(RESULT_DIR,"figs"))
