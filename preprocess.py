import numpy as np
import pandas as pd
import os
from config import FEATURE_DIR

IN = os.path.join(FEATURE_DIR, "features.parquet")
OUT = os.path.join(FEATURE_DIR, "features_clean.parquet")

print("Loading:", IN)
df = pd.read_parquet(IN)
print("Initial shape:", df.shape)

numeric_df = df.apply(lambda s: pd.to_numeric(s, errors='coerce'))

# numeric columns are those that have at least one non-null numeric parse
numeric_cols = numeric_df.columns[numeric_df.notnull().any()].tolist()
print("Detected numeric-like columns (count={}): {}".format(len(numeric_cols), numeric_cols))

# Replace original df's numeric-like columns with parsed numeric versions
df[numeric_cols] = numeric_df[numeric_cols]

# 2) Remove constant columns among numeric features (no variance)
non_const_numeric = [c for c in numeric_cols if df[c].nunique(dropna=True) > 1]
removed_consts = set(numeric_cols) - set(non_const_numeric)
if removed_consts:
    print("Dropping constant numeric columns:", sorted(removed_consts))

# Keep non-numeric columns (e.g., pcap_file, start_time may be numeric already)
other_cols = [c for c in df.columns if c not in non_const_numeric]
print("Non-numeric / reserved columns (kept):", other_cols)

# Build final DataFrame to work on: keep other_cols + non_const_numeric
df2 = pd.concat([df[other_cols], df[non_const_numeric]], axis=1)

print("After dropping constant numeric cols shape:", df2.shape)

# 3) Apply log1p to skewed numeric columns if present
for c in ['packet_count', 'total_bytes']:
    if c in df2.columns:
        # ensure non-negative before log1p; fillna with 0 for transformation
        df2[c] = np.log1p(df2[c].fillna(0).astype(float))

# 4) Drop rows (windows) that have >50% missing among numeric features
num_features = non_const_numeric
if len(num_features) == 0:
    raise RuntimeError("No numeric features detected to process. Aborting.")
missing_counts = df2[num_features].isna().sum(axis=1)
keep_mask = missing_counts <= 0.5 * len(num_features)
dropped_rows = (~keep_mask).sum()
print(f"Dropping {dropped_rows} rows with >50% missing numeric features (of {len(df2)} rows).")
df3 = df2.loc[keep_mask].copy()
print("Shape after dropping rows:", df3.shape)

# 5) Fill remaining missing numeric values with median (computed on remaining rows)
medians = df3[num_features].median()
df3[num_features] = df3[num_features].fillna(medians)

# 6) Final sanity: ensure numeric columns are numeric dtype
for c in num_features:
    df3[c] = pd.to_numeric(df3[c], errors='coerce')

# 7) Save cleaned features
print("Saving cleaned features to:", OUT)
df3.to_parquet(OUT, index=False)
print("Done. final shape:", df3.shape)
