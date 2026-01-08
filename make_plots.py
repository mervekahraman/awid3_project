import os, json, joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from vae import SimpleVAE
import torch

os.makedirs("results/figs", exist_ok=True)

scaler_pkg = joblib.load("models/scaler_awid.joblib")
features = scaler_pkg['features']
scaler = scaler_pkg['scaler']

df = pd.read_parquet("features/features_clean.parquet")
from config import NORMAL_PCAPS
test_df = df[~df['pcap_file'].isin(NORMAL_PCAPS)]  # all attack windows
# build test X
X_test = scaler.transform(test_df[features].values)

# IsolationForest PR-curve
iso = joblib.load("models/iso_model.joblib")['model']
scores_iso = -iso.decision_function(X_test)
# if you have labels for attacks (here test_df all attacks => y=1), but for PR need both classes:
normal_df = df[df['pcap_file'].isin(NORMAL_PCAPS)]
sample_norm = normal_df.sample(min(len(test_df), len(normal_df)), random_state=42)
mix_df = pd.concat([test_df, sample_norm])
y_true = mix_df['pcap_file'].apply(lambda x: 1 if x not in NORMAL_PCAPS else 0).values
X_mix = scaler.transform(mix_df[features].values)

scores_iso_mix = -iso.decision_function(X_mix)
prec, rec, _ = precision_recall_curve(y_true, scores_iso_mix)
pr_auc_iso = auc(rec, prec)
plt.figure(figsize=(6,4))
plt.plot(rec, prec)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"Iso PR curve (AUC={pr_auc_iso:.3f})")
plt.tight_layout()
plt.savefig("results/figs/pr_iso.png")
print("Saved results/figs/pr_iso.png")

# VAE recon error histogram (normal vs attack)
vae = SimpleVAE(input_dim=X_mix.shape[1], latent_dim=8)
vae.load_state_dict(torch.load("models/vae_awid.pth", map_location='cpu'))
vae.eval()
with torch.no_grad():
    Xt = torch.from_numpy(X_mix).float()
    recon, mu, logvar = vae(Xt)
    re = ((recon - Xt)**2).mean(dim=1).numpy()

plt.figure(figsize=(6,4))
plt.hist(re[y_true==0], bins=80, alpha=0.6, label="normal")
plt.hist(re[y_true==1], bins=80, alpha=0.6, label="attack")
plt.legend()
plt.title("VAE reconstruction error (normal vs attack)")
plt.tight_layout()
plt.savefig("results/figs/vae_recon_hist.png")
print("Saved results/figs/vae_recon_hist.png")
