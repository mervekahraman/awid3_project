import os, joblib, json, time, pandas as pd, numpy as np, traceback
from sklearn.metrics import precision_recall_curve, auc
from vae import SimpleVAE
import torch

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def safe_torch_load(path):

    try:
        obj = torch.load(path, map_location='cpu', weights_only=True)
        print(f"Loaded {path} with weights_only=True")
        return obj
    except Exception as e:
        print("weights_only load failed, falling back to full load with allowlist. Error:", e)
        try:
            torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
        except Exception as ee:
            print("add_safe_globals failed:", ee)
        obj = torch.load(path, map_location='cpu', weights_only=False)
        print(f"Loaded {path} with weights_only=False")
        return obj

def compute_pr_auc(y_true, scores):
    try:
        prec, rec, _ = precision_recall_curve(y_true, scores)
        return float(auc(rec, prec))
    except Exception:
        return None

def timed_latency(func, repeat=100):
    t0 = time.time()
    for i in range(repeat):
        try:
            func()
        except Exception:
            pass
    t1 = time.time()
    return (t1 - t0) / repeat * 1000.0

models_info = []

# load scaler & features
scaler_pkg = joblib.load("models/scaler_awid.joblib")
scaler = scaler_pkg['scaler']
feats = scaler_pkg['features']
print("Features used:", feats)

# load cleaned AWID feature dataframe
df = pd.read_parquet("features/features_clean.parquet")
from config import NORMAL_PCAPS

# build mixed test set (attack windows = not in NORMAL_PCAPS)
test_df = df[~df['pcap_file'].isin(NORMAL_PCAPS)]
normal_df = df[df['pcap_file'].isin(NORMAL_PCAPS)]
if normal_df.shape[0] == 0:
    print("Warning: NORMAL_PCAPS yielded zero rows; using whole df as normal sample.")
    normal_df = df.copy()

# match sizes for PR evaluation
n_test = max(1, len(test_df))
sample_norm = normal_df.sample(min(len(normal_df), n_test), random_state=42)
mix_df = pd.concat([test_df, sample_norm], ignore_index=True)
y_true = mix_df['pcap_file'].apply(lambda x: 1 if x not in NORMAL_PCAPS else 0).values
X_mix = scaler.transform(mix_df[feats].fillna(0).values)
print("Evaluation mix shape:", X_mix.shape)

# IsolationForest
try:
    iso_pkg = joblib.load("models/iso_model.joblib")
    iso = iso_pkg.get('model', iso_pkg)
    scores_iso = -iso.decision_function(X_mix)
    pr_iso = compute_pr_auc(y_true, scores_iso)
    size_iso = os.path.getsize("models/iso_model.joblib")/1024**2 if os.path.exists("models/iso_model.joblib") else None
    lat_iso_ms = timed_latency(lambda: iso.decision_function(X_mix[:1]))
    models_info.append({"model":"IsolationForest","pr_auc":pr_iso,"size_mb":size_iso,"latency_ms":lat_iso_ms})
    print("Iso PR-AUC:", pr_iso)
except Exception as e:
    print("IsolationForest eval failed:", e, traceback.format_exc())

# OneClassSVM
try:
    oc_pkg = joblib.load("models/ocsvm_model.joblib")
    oc = oc_pkg.get('model', oc_pkg)
    scores_oc = -oc.score_samples(X_mix)
    pr_oc = compute_pr_auc(y_true, scores_oc)
    size_oc = os.path.getsize("models/ocsvm_model.joblib")/1024**2 if os.path.exists("models/ocsvm_model.joblib") else None
    lat_oc_ms = timed_latency(lambda: oc.score_samples(X_mix[:1]))
    models_info.append({"model":"OneClassSVM","pr_auc":pr_oc,"size_mb":size_oc,"latency_ms":lat_oc_ms})
    print("OCSVM PR-AUC:", pr_oc)
except Exception as e:
    print("OneClassSVM eval failed:", e, traceback.format_exc())

# VAE
try:
    vae = SimpleVAE(input_dim=X_mix.shape[1], latent_dim=8)
    vae.load_state_dict(torch.load("models/vae_awid.pth", map_location='cpu'))
    vae.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X_mix).float()
        recon, mu, logvar = vae(Xt)
        re = ((recon - Xt)**2).mean(dim=1).numpy()
    pr_vae = compute_pr_auc(y_true, re)
    size_vae = os.path.getsize("models/vae_awid.pth")/1024**2 if os.path.exists("models/vae_awid.pth") else None
    lat_vae_ms = timed_latency(lambda: vae(torch.from_numpy(X_mix[:1]).float()))
    models_info.append({"model":"VAE","pr_auc":pr_vae,"size_mb":size_vae,"latency_ms":lat_vae_ms})
    print("VAE PR-AUC:", pr_vae)
except Exception as e:
    print("VAE eval failed:", e, traceback.format_exc())

# Deep-SVDD (best-effort)
ds_metrics = {"model":"DeepSVDD","pr_auc":None,"size_mb":None,"latency_ms":None}
try:
    p = "models/deepsvdd.pth"
    if not os.path.exists(p):
        raise FileNotFoundError(p + " not found")
    ds = safe_torch_load(p)  # safe_torch_load fonksiyonun zaten tanımlı olduğunu varsayıyoruz
    print("Deep-SVDD checkpoint keys:", list(ds.keys()) if isinstance(ds, dict) else type(ds))

    # get center if present
    center = None
    if isinstance(ds, dict):
        center = ds.get('center', None)
        if isinstance(center, list):
            center = np.array(center)
        elif isinstance(center, torch.Tensor):
            center = center.cpu().numpy()

    # try to get state_dict (encoder weights)
    state = None
    if isinstance(ds, dict):
        # prefer 'state_dict' key, else scan for a dict-like entry
        if 'state_dict' in ds and isinstance(ds['state_dict'], dict):
            state = ds['state_dict']
        else:
            for k,v in ds.items():
                if isinstance(v, dict) and any(isinstance(x, (torch.Tensor, np.ndarray)) for x in v.values()):
                    state = v
                    break

    if state is None:
        print("No state_dict-like object found in checkpoint; skipping DeepSVDD eval.")
        ds_metrics['size_mb'] = os.path.getsize(p)/1024**2
    else:
        zdim = None
        if center is not None:
            zdim = int(np.asarray(center).shape[0])
        # load Encoder class if available
        try:
            from train_deepsvdd import Encoder as DS_Encoder
            have_enc_class = True
        except Exception as e:
            print("train_deepsvdd.Encoder class not importable:", e)
            have_enc_class = False

        if have_enc_class:
            # input dim = number of features
            inp_dim = len(feats)
            if zdim is None:
                # try to infer zdim from state dict keys shapes (look for fc layers)
                for kk, vv in state.items():
                    try:
                        shape = vv.shape if hasattr(vv, 'shape') else None
                        if shape is not None and len(shape) == 2 and shape[1] < 128:
                            # heuristic: last linear layer weight matrix maybe (z_dim)
                            zdim = shape[0]
                            break
                    except Exception:
                        pass
                if zdim is None:
                    zdim = 8  # fallback
            enc = DS_Encoder(inp_dim, z_dim=zdim)
            # try direct load first
            loaded_enc = False
            try:
                enc.load_state_dict(state)
                loaded_enc = True
                print("Loaded state_dict directly into Encoder.")
            except Exception as e:
                print("Direct load into Encoder failed:", e)
                # try stripping common prefixes
                new_state = {}
                for k, v in state.items():
                    kk = k
                    if kk.startswith("encoder."):
                        kk = kk[len("encoder."):]
                    if kk.startswith("net."):
                        kk = kk[len("net."):]
                    if kk.startswith("model."):
                        kk = kk[len("model."):]
                    if kk.startswith("module."):
                        kk = kk[len("module."):]
                    new_state[kk] = v
                try:
                    enc.load_state_dict(new_state)
                    loaded_enc = True
                    print("Loaded state_dict after prefix stripping.")
                except Exception as e2:
                    print("Prefix-stripped load also failed:", e2)

            if loaded_enc:
                # compute latent distances
                enc.eval()
                import pandas as pd
                X_all = pd.read_parquet("features/features_clean.parquet")
                test_df = X_all[~X_all['pcap_file'].isin(NORMAL_PCAPS)]
                normal_df = X_all[X_all['pcap_file'].isin(NORMAL_PCAPS)]
                if normal_df.shape[0] == 0:
                    normal_df = X_all.copy()
                sample_norm = normal_df.sample(min(len(test_df), len(normal_df)), random_state=42)
                mix_df = pd.concat([test_df, sample_norm], ignore_index=True)
                X_mix = scaler.transform(mix_df[feats].fillna(0).values)

                with torch.no_grad():
                    zv = enc(torch.from_numpy(X_mix).float())
                    c = torch.tensor(center).float() if center is not None else torch.zeros(zv.shape[1])
                    dists = ((zv - c)**2).sum(dim=1).numpy()

                pr_ds = compute_pr_auc(mix_df['pcap_file'].apply(lambda x: 1 if x not in NORMAL_PCAPS else 0).values, dists)
                size_ds = os.path.getsize(p)/1024**2
                lat_ds = timed_latency(lambda: enc(torch.from_numpy(X_mix[:1]).float()))
                ds_metrics.update({"pr_auc":pr_ds,"size_mb":size_ds,"latency_ms":lat_ds})
                print("DeepSVDD PR-AUC computed:", pr_ds)
            else:
                print("Could not load encoder weights into Encoder class — skipping DeepSVDD evaluation.")
                ds_metrics['size_mb'] = os.path.getsize(p)/1024**2
        else:
            print("No Encoder class available to evaluate DeepSVDD — skipping.")
            ds_metrics['size_mb'] = os.path.getsize(p)/1024**2

except Exception as e:
    print("DeepSVDD section failed:", e)
    import traceback as _tb
    print(_tb.format_exc())

# append ds metrics if pr computed or at least size known
if ds_metrics.get('pr_auc') is not None or ds_metrics.get('size_mb') is not None:
    models_info.append(ds_metrics)
