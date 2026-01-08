import pandas as pd, joblib, torch, os
from vae import SimpleVAE
from config import NORMAL_PCAPS
pkg = joblib.load("models/scaler_awid.joblib")
scaler = pkg['scaler']; feats = pkg['features']
df = pd.read_parquet("features/features_clean.parquet")  # kullanılan temiz df
X = scaler.transform(df[feats].fillna(0).values)
vae = SimpleVAE(input_dim=X.shape[1], latent_dim=8)
vae.load_state_dict(torch.load("models/vae_awid.pth", map_location='cpu'))
vae.eval()
import numpy as np
with torch.no_grad():
    recon, _, _ = vae(torch.from_numpy(X).float())
    re = ((recon - torch.from_numpy(X).float())**2).mean(dim=1).numpy()
df['vae_recon'] = re
norm = df[df['pcap_file'].isin(NORMAL_PCAPS)]['vae_recon']
att = df[~df['pcap_file'].isin(NORMAL_PCAPS)]['vae_recon']
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.hist(norm, bins=80, alpha=0.6, label='normal')
plt.hist(att, bins=80, alpha=0.6, label='attack')
plt.legend(); plt.title("VAE recon error (normal vs attack)")
os.makedirs("results/figs", exist_ok=True)
plt.tight_layout()
plt.savefig("results/figs/vae_recon_dist.png")
plt.close()
print("Saved VAE recon distributions.")
