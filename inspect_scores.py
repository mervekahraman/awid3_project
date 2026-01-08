import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

R = Path("results/threshold_tuning/reports")
files = list(R.glob("*_ytrue_ypred.csv"))
for f in files:
    df = pd.read_csv(f)
    model = f.stem.replace("_ytrue_ypred","")
    print(model, "shape", df.shape, "scores min/max", df['score'].min(), df['score'].max())
    plt.figure(figsize=(6,3))
    plt.hist(df['score'], bins=80)
    plt.title(f"{model} score histogram")
    plt.xlabel("score"); plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(R / f"{model}_score_hist.png")
    plt.close()
print("Saved histograms to", R)
