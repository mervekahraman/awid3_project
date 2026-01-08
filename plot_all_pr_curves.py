# plot_all_pr_curves.py
import os, joblib, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import torch
from vae import SimpleVAE

os.makedirs("results/figs", exist_ok=True)

pkg = joblib.load("models/scaler_awid.joblib")
scaler = pkg['scaler']; feats = pkg['features']
df = pd.read_parquet("features/features_clean.parquet")
from config import NORMAL_PCAPS
test_df = df[~df['pcap_file'].isin(NORMAL_PCAPS)]
normal_df = df[df['pcap_file'].isin(NORMAL_PCAPS)]
if normal_df.shape[0] == 0:
    normal_df = df.copy()
sample_norm = normal_df.sample(min(len(normal_df), len(test_df)), random_state=42)
mix_df = pd.concat([test_df, sample_norm], ignore_index=True)
y_true = mix_df['pcap_file'].apply(lambda x: 1 if x not in NORMAL_PCAPS else 0).values
X_mix = scaler.transform(mix_df[feats].fillna(0).values)

# helper to plot one curve
def plot_pr(y, scores, label, ax):
    prec, rec, _ = precision_recall_curve(y, scores)
    ax.plot(rec, prec, label=f"{label} (AUC={auc(rec,prec):.3f})")
    return auc(rec,prec)

fig, ax = plt.subplots(figsize=(7,6))

# IsolationForest
iso = joblib.load("models/iso_model.joblib")['model']
scores_iso = -iso.decision_function(X_mix)
auc_iso = plot_pr(y_true, scores_iso, "IsolationForest", ax)

# OneClassSVM
oc = joblib.load("models/ocsvm_model.joblib")['model']
scores_oc = -oc.score_samples(X_mix)
auc_oc = plot_pr(y_true, scores_oc, "OneClassSVM", ax)

# VAE
vae = SimpleVAE(input_dim=X_mix.shape[1], latent_dim=8)
vae.load_state_dict(torch.load("models/vae_awid.pth", map_location='cpu'))
vae.eval()
with torch.no_grad():
    Xt = torch.from_numpy(X_mix).float()
    recon, mu, logvar = vae(Xt)
    re = ((recon - Xt)**2).mean(dim=1).numpy()
auc_vae = plot_pr(y_true, re, "VAE (recon_error)", ax)

# DeepSVDD (if available)
try:
    ds = torch.load("models/deepsvdd.pth", map_location='cpu', weights_only=False)
    state = ds.get('state_dict', None)
    center = ds.get('center', None)
    if isinstance(center, list): center = np.array(center)
    # load encoder (assumes train_deepsvdd.Encoder present)
    from train_deepsvdd import Encoder
    enc = Encoder(len(feats), z_dim=center.shape[0])
    enc.load_state_dict(state)
    enc.eval()
    with torch.no_grad():
        zv = enc(torch.from_numpy(X_mix).float())
        dists = ((zv - torch.tensor(center).float())**2).sum(dim=1).numpy()
    auc_ds = plot_pr(y_true, dists, "DeepSVDD (dist)", ax)
except Exception as e:
    print("DeepSVDD plotting skipped:", e)
    auc_ds = None

ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("PR curves (AWID-based evaluation mix)")
ax.legend()
plt.tight_layout()
plt.savefig("results/figs/pr_curve_all.png", dpi=150)
print("Saved results/figs/pr_curve_all.png")

# bar of PR-AUC
labels = ["IsolationForest","OneClassSVM","VAE","DeepSVDD"]
aucs = [auc_iso, auc_oc, auc_vae, auc_ds if auc_ds is not None else 0]
plt.figure(figsize=(6,4))
plt.bar(labels, [x if x is not None else 0 for x in aucs])
plt.ylim(0,1)
plt.ylabel("PR-AUC")
plt.tight_layout()
plt.savefig("results/figs/pr_auc_bar.png", dpi=150)
print("Saved results/figs/pr_auc_bar.png")
