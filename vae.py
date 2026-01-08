import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib
import numpy as np

class SimpleVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.fc_dec1 = nn.Linear(latent_dim, 32)
        self.fc_dec2 = nn.Linear(32, 64)
        self.fc_out = nn.Linear(64, input_dim)
        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.relu(self.fc_dec1(z))
        h = self.relu(self.fc_dec2(h))
        return self.fc_out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar

def loss_fn(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld, recon_loss.item(), kld.item()

def train_vae(X, epochs=50, batch_size=64, latent_dim=8, lr=1e-3, device='cpu'):
    X = np.array(X, dtype=np.float32)
    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SimpleVAE(input_dim=X.shape[1], latent_dim=latent_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        total_loss = 0.0
        for batch in loader:
            x = batch[0].to(device)
            recon, mu, logvar = model(x)
            loss, rl, kld = loss_fn(recon, x, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        print(f"Epoch {ep+1}/{epochs} Loss: {total_loss/len(dataset):.6f}")
    return model

if __name__ == "__main__":
    import pandas as pd
    from config import FEATURE_OUTPUT
    df = pd.read_parquet(FEATURE_OUTPUT)
    features = ['packet_count','total_bytes','avg_pkt_len','unique_src','unique_dst','mean_interarrival','rssi_mean','rssi_std']
    X = df[features].values
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = train_vae(Xs, epochs=25, batch_size=128, latent_dim=8, device='cpu')
    joblib.dump({'vae': model.state_dict(), 'scaler': scaler, 'features': features}, "models/vae_model.joblib")
    print("VAE model kaydedildi.")
