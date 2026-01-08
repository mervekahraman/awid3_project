import torch, os
import numpy as np

MODEL_DIR = "models"
src = os.path.join(MODEL_DIR, "deepsvdd.pth")
dst = os.path.join(MODEL_DIR, "deepsvdd_safe.pth")

# allowlist needed global to be able to load original
torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])

print("Loading original checkpoint:", src)
ck = torch.load(src, map_location='cpu', weights_only=False)
print("Loaded keys:", list(ck.keys()))

# Convert any numpy arrays to Python lists
new_ck = {}
for k, v in ck.items():
    if isinstance(v, np.ndarray):
        new_ck[k] = v.tolist()
    else:
        # if it's a torch state_dict (OrderedDict), keep as-is
        new_ck[k] = v

# Save back: use torch.save (state_dict still fine; center becomes list)
torch.save(new_ck, dst)
print("Saved converted checkpoint to:", dst)
