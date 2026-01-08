import torch, joblib, numpy as np, traceback
from train_deepsvdd import Encoder
from pathlib import Path
p = "models/deepsvdd.pth"
torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
ck = torch.load(p, map_location='cpu', weights_only=False)
print("keys:", list(ck.keys()))
state = ck.get('state_dict', None)
center = ck.get('center', None)
if state is None:
    print("No state_dict found; cannot load encoder automatically.")
    raise SystemExit
import numpy as np
center_arr = np.array(center) if center is not None else None
zdim = center_arr.shape[0] if center_arr is not None else 8
print("Detected latent dim:", zdim)
# build encoder with feats dim detection
scaler_pkg = joblib.load("models/scaler_awid.joblib")
feats = scaler_pkg['features']
inp_dim = len(feats)
enc = Encoder(inp_dim, z_dim=zdim)
# try to adapt key prefixes: many implementations save under 'net.' or 'model.'
sd_keys = list(state.keys())
# direct load?
try:
    enc.load_state_dict(state)
    print("Loaded state_dict directly into Encoder.")
except Exception as e:
    print("Direct load failed:", e)
    # try filtering keys with 'encoder' in name
    new_state = {}
    for k,v in state.items():
        kk = k
        if kk.startswith("encoder."):
            kk = kk[len("encoder."):]
        # try other common prefixes
        if kk.startswith("net."):
            kk = kk[len("net."):]
        if kk.startswith("model."):
            kk = kk[len("model."):]
        new_state[kk] = v
    try:
        enc.load_state_dict(new_state)
        print("Loaded state_dict after prefix stripping.")
    except Exception as e2:
        print("Prefix stripping failed:", e2)
        print("Available keys sample:", sd_keys[:30])
        raise

df = joblib.load("models/scaler_awid.joblib")
import pandas as pd
X = pd.read_parquet("features/features_clean.parquet")
from config import NORMAL_PCAPS
test_df = X[~X['pcap_file'].isin(NORMAL_PCAPS)]
normal_df = X[X['pcap_file'].isin(NORMAL_PCAPS)].sample(min(len(test_df), len(X)), random_state=42)
mix_df = pd.concat([test_df, normal_df])
scaler = joblib.load("models/scaler_awid.joblib")['scaler']
feats = joblib.load("models/scaler_awid.joblib")['features']
X_mix = scaler.transform(mix_df[feats].fillna(0).values)
enc.eval()
with torch.no_grad():
    zv = enc(torch.from_numpy(X_mix).float())
    c = torch.tensor(center_arr).float()
    dists = ((zv - c)**2).sum(dim=1).numpy()
from sklearn.metrics import precision_recall_curve, auc
y = mix_df['pcap_file'].apply(lambda x: 1 if x not in NORMAL_PCAPS else 0).values
print("DeepSVDD PR-AUC:", auc(*precision_recall_curve(y, dists)[:2][::-1]))
