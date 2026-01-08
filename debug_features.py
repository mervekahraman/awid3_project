import csv, glob, sys
from pathlib import Path
import pandas as pd
import numpy as np

def detect_sep(file_path, sample_bytes=8192):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(sample_bytes)
    try:
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
    except Exception:
        return ','

def parse_rssi(val):
    if not isinstance(val, str):
        return np.nan
    v = val.replace("\\", "")
    parts = v.split(",")
    nums = []
    for p in parts:
        try:
            nums.append(float(p))
        except:
            pass
    return float(np.mean(nums)) if nums else np.nan

def inspect_file(fp, nrows_sample=200000):
    fp = Path(fp)
    print("\n\n=== FILE:", fp.name, "===\n")
    sep = detect_sep(fp)
    print("detected sep:", repr(sep))

    # read a sample chunk (nrows) as strings - tolerant read
    try:
        df = pd.read_csv(fp, sep=sep, nrows=nrows_sample, engine='python', dtype=str, on_bad_lines='skip', encoding='utf-8')
    except Exception as e:
        print("read_csv failed:", e)
        return

    print("Columns:", df.columns.tolist())
    print("Sample rows read:", len(df))
    if len(df)==0:
        print("-> Dosyadan 0 satır okundu.")
        return


    candidates = [c for c in df.columns if 'time' in c.lower()]
    print("time candidates:", candidates)

    for c in candidates:
        parsed = pd.to_numeric(df[c], errors='coerce')
        nonnull = parsed.notnull().sum()
        print(f"  Col `{c}` -> parsed non-null: {nonnull}/{len(parsed)}")
        if nonnull>0:
            vals = parsed.dropna().astype(float)
            print(f"    min={vals.min():.6f} max={vals.max():.6f} sample head:", list(vals.head(10).values))

    # If frame.time_epoch exists, show conversion
    if 'frame.time_epoch' in df.columns:
        t = pd.to_numeric(df['frame.time_epoch'], errors='coerce')
        print("frame.time_epoch non-null:", t.notnull().sum(), "first 10:", list(t.dropna().head(10).values))

    # check radiotap examples
    if 'radiotap.dbm_antsignal' in df.columns:
        sample_rssi = df['radiotap.dbm_antsignal'].dropna().head(10).tolist()
        print("radiotap.dbm_antsignal sample (raw):", sample_rssi)
        parsed = [parse_rssi(x) for x in sample_rssi]
        print("radiotap parsed sample (mean):", parsed)
    else:
        print("radiotap.dbm_antsignal not in columns")

    # attempt window creation using detected time col (choose best candidate)
    chosen = None
    for c in candidates:
        parsed = pd.to_numeric(df[c], errors='coerce')
        if parsed.notnull().sum() > 0:
            chosen = c
            break
    if chosen is None:
        print("No usable time column found in sample -> this file will produce 0 features.")
        return

    df['time_num'] = pd.to_numeric(df[chosen], errors='coerce')
    tmin = float(df['time_num'].dropna().min())
    print("tmin:", tmin)
    # compute window_id for sample rows
    df_valid = df[df['time_num'].notnull()].copy()
    df_valid['window_id'] = ((df_valid['time_num'] - tmin) // 10.0).astype(int)
    print("Valid rows after time filter:", len(df_valid))
    print("Unique window ids in sample:", df_valid['window_id'].nunique())
    print("Window id value counts (top 10):")
    print(df_valid['window_id'].value_counts().head(10))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        files = [sys.argv[1]]
    else:
        files = glob.glob("csv_exports/*.csv")
        # prefer one problematic file first:
        files = sorted(files)

    for i, f in enumerate(files[:3]):
        inspect_file(f)
