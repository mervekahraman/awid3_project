import pandas as pd
from config import FEATURE_DIR

fn = FEATURE_DIR + "/features.parquet"
print("Checking:", fn)
df = pd.read_parquet(fn)
pcaps = sorted(df['pcap_file'].unique().tolist())
print("Found pcap files (count={}):".format(len(pcaps)))
for p in pcaps:
    print(" -", p)
