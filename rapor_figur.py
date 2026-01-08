import os, joblib, time
models = ["models/iso_model.joblib","models/ocsvm_model.joblib","models/vae_awid.pth","models/deepsvdd.pth"]
for m in models:
    try:
        size_mb = os.path.getsize(m)/1024**2
    except:
        size_mb = None
    print(m, "size_MB=", size_mb)
iso = joblib.load("models/iso_model.joblib")['model']
import numpy as np
Xs = np.random.randn(100, len(joblib.load("models/scaler_awid.joblib")['features']))
t0 = time.time()
for i in range(100):
    _ = iso.decision_function(Xs[:1])
t1 = time.time()
print("iso latency avg ms:", (t1-t0)/100*1000)
