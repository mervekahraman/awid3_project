import os, joblib, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from vae import SimpleVAE, loss_fn
from sklearn.preprocessing import StandardScaler
from config import FEATURE_DIR, MODEL_DIR
from pathlib import Path

pkg = joblib.load(os.path.join(MODEL_DIR, "scaler_awid.joblib"))
scaler = pkg['scaler']
features = pkg['features']

df = pd.read_parquet(os.path.join(FEATURE_DIR, "features_clean.parquet"))
# train on normal windows
from config import NORMAL_PCAPS
train_df = df[df['pcap_file'].isin(NORMAL_PCAPS)]
# split train/val
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

X_train = scaler.transform(train_df[features].values).astype(np.float32)
X_val = scaler.transform(val_df[features].values).astype(np.float32)

bs = 256
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train)), batch_size=bs, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val)), batch_size=bs, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleVAE(input_dim=X_train.shape[1], latent_dim=8).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
best_val = 1e9
patience = 10
pat_cnt = 0
epochs = 100

for ep in range(1, epochs+1):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        x = batch[0].to(device)
        recon, mu, logvar = model(x)
        loss, rl, kld = loss_fn(recon, x, mu, logvar)
        opt.zero_grad(); loss.backward(); opt.step()
        train_loss += loss.item() * x.size(0)
    train_loss = train_loss / len(train_loader.dataset)

    # val
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            recon, mu, logvar = model(x)
            loss, rl, kld = loss_fn(recon, x, mu, logvar)
            val_loss += loss.item() * x.size(0)
    val_loss = val_loss / len(val_loader.dataset)

    print(f"Epoch {ep} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "vae_awid.pth"))
        pat_cnt = 0
    else:
        pat_cnt += 1
        if pat_cnt >= patience:
            print("Early stopping.")
            break

joblib.dump({'features':features}, os.path.join(MODEL_DIR, "vae_meta.joblib"))
print("VAE training done. Model saved.")
