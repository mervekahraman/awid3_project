import os, joblib, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from config import FEATURE_DIR, MODEL_DIR
import pandas as pd

pkg = joblib.load(os.path.join(MODEL_DIR, "scaler_awid.joblib"))
scaler = pkg['scaler']; features = pkg['features']

df = pd.read_parquet(os.path.join(FEATURE_DIR, "features_clean.parquet"))
from config import NORMAL_PCAPS
train_df = df[df['pcap_file'].isin(NORMAL_PCAPS)]
X = scaler.transform(train_df[features].values).astype('float32')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, in_dim, z_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, z_dim)
        )
    def forward(self, x): return self.net(x)

enc = Encoder(X.shape[1], z_dim=8).to(device)
opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=256, shuffle=True)

# initialize center c as mean of embeddings
with torch.no_grad():
    emb = enc(torch.from_numpy(X).to(device)).cpu().numpy()
c = emb.mean(axis=0)
c = torch.tensor(c, dtype=torch.float32, device=device)

# train to minimize ||z-c||^2
for epoch in range(60):
    enc.train()
    tot=0.0
    for b in loader:
        x = b[0].to(device)
        z = enc(x)
        loss = ((z - c)**2).sum(dim=1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()*x.size(0)
    print("Epoch", epoch+1, "loss", tot/len(loader.dataset))
torch.save({'state_dict':enc.state_dict(), 'center': c.cpu().numpy(), 'features':features}, os.path.join(MODEL_DIR, "deepsvdd.pth"))
print("Saved Deep-SVDD model.")
