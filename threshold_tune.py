import os
import json
import joblib
import numpy as np
import pandas as pd
from config import FEATURE_DIR, MODEL_DIR, NORMAL_PCAPS
from sklearn.metrics import precision_recall_curve, auc
import torch

# load features & scaler
df = pd.read_parquet(os.path.join(FEATURE_DIR, "features_clean.parquet"))
pkg = joblib.load(os.path.join(MODEL_DIR, "scaler_awid.joblib"))
scaler = pkg['scaler']
features = pkg['features']

# split normal windows to get validation set
from sklearn.model_selection import train_test_split
normal_df = df[df['pcap_file'].isin(NORMAL_PCAPS)]
train_df, val_df = train_test_split(normal_df, test_size=0.2, random_state=42)

# prepare X validation
Xv = scaler.transform(val_df[features].fillna(0).values)

results = {}

# ensure results dir
os.makedirs("results", exist_ok=True)

# IsolationForest
iso_pkg = joblib.load(os.path.join(MODEL_DIR, "iso_model.joblib"))
iso = iso_pkg['model']
scores_iso = -iso.decision_function(Xv)
thr_iso = float(np.percentile(scores_iso, 99))  # 99th percentile rule
results['iso'] = {'threshold': thr_iso}
print("[iso] thr:", thr_iso, "scores min/max:", float(scores_iso.min()), float(scores_iso.max()))

# One-Class SVM
oc_pkg = joblib.load(os.path.join(MODEL_DIR, "ocsvm_model.joblib"))
oc = oc_pkg['model']
scores_oc = -oc.score_samples(Xv)
thr_oc = float(np.percentile(scores_oc, 99))
results['ocsvm'] = {'threshold': thr_oc}
print("[ocsvm] thr:", thr_oc, "scores min/max:", float(scores_oc.min()), float(scores_oc.max()))

# VAE: compute reconstruction errors using saved VAE
from vae import SimpleVAE
vae_path = os.path.join(MODEL_DIR, "vae_awid.pth")
vae = SimpleVAE(input_dim=Xv.shape[1], latent_dim=8)
try:
    vae.load_state_dict(torch.load(vae_path, map_location='cpu'))
    vae.eval()
    Xv_t = torch.from_numpy(Xv).float()
    with torch.no_grad():
        recon, mu, logvar = vae(Xv_t)
        re = ((recon - Xv_t)**2).mean(dim=1).numpy()
    thr_vae = float(np.percentile(re, 99))
    results['vae'] = {'threshold': thr_vae}
    print("[vae] thr:", thr_vae, "re min/max:", float(re.min()), float(re.max()))
except Exception as e:
    print("VAE load/compute error:", e)
    results['vae'] = {'threshold': None, 'error': str(e)}

# Deep-SVDD: robust load with fallback for torch.weights_only issues
deepsvdd_path = os.path.join(MODEL_DIR, "deepsvdd_safe.pth")
if not os.path.exists(deepsvdd_path):
    # try the other filename used in repo
    deepsvdd_path = os.path.join(MODEL_DIR, "deepsvdd.pth")

if os.path.exists(deepsvdd_path):
    try:
        # try safe weights_only load first
        ds = torch.load(deepsvdd_path, map_location='cpu', weights_only=True)
        print("Loaded deepsvdd with weights_only=True")
    except Exception as e1:
        print("weights_only load failed, trying full load with allowlist. Error:", e1)
        # allow numpy reconstruct if needed (only do for your trusted checkpoint)
        try:
            import numpy as _np
            torch.serialization.add_safe_globals([_np._core.multiarray._reconstruct])
        except Exception:
            pass
        ds = torch.load(deepsvdd_path, map_location='cpu', weights_only=False)
        print("Loaded deepsvdd with weights_only=False (full load)")

    # ds expected to contain 'state_dict' and 'center'
    thr_ds = None
    try:
        center = ds.get('center', None)
        if center is None:
            raise RuntimeError("deepsvdd checkpoint has no 'center' field")
        center_t = torch.tensor(np.array(center)).float()

        # import Encoder from train_deepsvdd (ensure same class definition)
        try:
            from train_deepsvdd import Encoder
        except Exception as ie:
            raise RuntimeError("Cannot import Encoder from train_deepsvdd.py: " + str(ie))

        enc = Encoder(input_dim=Xv.shape[1], z_dim=8)  # keep z_dim=8 (as used earlier)
        # load state_dict if present
        if 'state_dict' in ds:
            try:
                enc.load_state_dict(ds['state_dict'])
                print("Loaded Encoder state_dict into encoder.")
            except Exception as le:
                # maybe state_dict is nested; try to load keys that match
                try:
                    enc.load_state_dict(ds['state_dict'], strict=False)
                    print("Loaded Encoder with strict=False.")
                except Exception as le2:
                    raise RuntimeError("Failed to load encoder state_dict: " + str(le2))
        else:
            raise RuntimeError("deepsvdd checkpoint missing 'state_dict'")

        enc.eval()
        with torch.no_grad():
            zv = enc(torch.from_numpy(Xv).float())
            dists = ((zv - center_t)**2).sum(dim=1).numpy()
        thr_ds = float(np.percentile(dists, 99))
        results['deepsvdd'] = {'threshold': thr_ds}
        print("[deepsvdd] thr:", thr_ds, "dists min/max:", float(dists.min()), float(dists.max()))
    except Exception as ee:
        print("Deep-SVDD scoring error:", ee)
        results['deepsvdd'] = {'threshold': None, 'error': str(ee)}
else:
    print("No deepsvdd checkpoint found at expected paths.")
    results['deepsvdd'] = {'threshold': None, 'error': 'checkpoint not found'}

# write results
out_path = os.path.join("results", "thresholds.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print("Saved thresholds to", out_path)
print(results)