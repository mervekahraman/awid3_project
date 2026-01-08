import os, glob, csv
from pathlib import Path
import numpy as np
import pandas as pd
from config import CIC_CSV_DIR, CIC_FEATURE_DIR, WINDOW_SEC, CSV_CHUNKSIZE
from utils import ensure_dirs

ensure_dirs([CIC_FEATURE_DIR])

OUT_CSV = os.path.join(CIC_FEATURE_DIR, "cic_mapped.csv")
OUT_PARQUET = os.path.join(CIC_FEATURE_DIR, "cic_mapped.parquet")

if os.path.exists(OUT_CSV):
    os.remove(OUT_CSV)

header = ['pcap_file','window_id','start_time','packet_count','total_bytes','avg_pkt_len',
          'unique_src','unique_dst','mean_interarrival','rssi_mean','rssi_std']
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(header)

def find_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def parse_time_series(s: pd.Series):

    tnum = pd.to_numeric(s.astype(str).str.strip(), errors='coerce')
    if tnum.notnull().any():
        # numeric values likely already epoch seconds or floats
        return tnum.astype(float)
    # try datetime parse
    td = pd.to_datetime(s, errors='coerce', infer_datetime_format=True)
    if td.notnull().any():
        # convert to epoch seconds (float)
        return td.astype('int64') / 1e9
    # fallback: all NaN
    return pd.Series([np.nan]*len(s), index=s.index)

def process_file(fp, chunksize=CSV_CHUNKSIZE):
    fname = os.path.basename(fp)
    print(f"\nProcessing CIC file: {fname}")
    # read small header to detect columns
    sample = pd.read_csv(fp, nrows=5)
    cols = sample.columns.tolist()

    # column mapping based on your uploaded headers
    time_col = find_col(cols, ['Timestamp', 'Time', 'timestamp', 'ts'])
    fwd_pkts = find_col(cols, ['Tot Fwd Pkts', 'Tot Fwd Packets', 'Tot Fwd Pkts'])
    bwd_pkts = find_col(cols, ['Tot Bwd Pkts', 'Tot Bwd Packets', 'Tot Bwd Pkts'])
    fwd_bytes = find_col(cols, ['TotLen Fwd Pkts', 'TotLen Fwd Packets', 'TotLen Fwd Pkts'])
    bwd_bytes = find_col(cols, ['TotLen Bwd Pkts', 'TotLen Bwd Packets', 'TotLen Bwd Pkts'])
    pkt_len_mean = find_col(cols, ['Pkt Len Mean', 'Pkt Size Avg', 'Pkt Len Mean '])
    flow_iat_mean = find_col(cols, ['Flow IAT Mean', 'Flow IAT Mean '])

    it = pd.read_csv(fp, chunksize=chunksize, low_memory=False, dtype=str)
    first_chunk = True
    tmin_file = None
    rows_written = 0

    for chunk_idx, chunk in enumerate(it):
        # ensure columns trimmed
        chunk.columns = [c.strip() for c in chunk.columns]

        if time_col not in chunk.columns:
            print(f"  SKIP chunk {chunk_idx}: no time column '{time_col}' present in {fname}")
            continue

        # parse time robustly
        chunk_time_parsed = parse_time_series(chunk[time_col])
        chunk['time_parsed'] = chunk_time_parsed

        # drop rows without parsed time
        before = len(chunk)
        chunk = chunk[chunk['time_parsed'].notnull()].copy()
        after = len(chunk)
        if after == 0:
            print(f"  chunk #{chunk_idx}: no parsable timestamps ({before} rows -> 0 after).")
            continue

        if first_chunk:
            tmin_file = float(chunk['time_parsed'].min())
            first_chunk = False

        # compute window id relative to tmin_file
        chunk['window_id'] = ((chunk['time_parsed'].astype(float) - tmin_file) // WINDOW_SEC).astype(int)
        chunk['start_time'] = (tmin_file + chunk['window_id'] * WINDOW_SEC).astype(float)

        # parse numeric fields safely
        def to_num(colname, default=0.0):
            if colname and colname in chunk.columns:
                return pd.to_numeric(chunk[colname].astype(str).str.replace(r'[^0-9\.\-]', '', regex=True), errors='coerce').fillna(default)
            else:
                return pd.Series([default]*len(chunk), index=chunk.index)

        chunk['fwd_pkts_num'] = to_num(fwd_pkts, 0).astype(int)
        chunk['bwd_pkts_num'] = to_num(bwd_pkts, 0).astype(int)
        chunk['fwd_bytes_num'] = to_num(fwd_bytes, 0.0)
        chunk['bwd_bytes_num'] = to_num(bwd_bytes, 0.0)
        chunk['pkt_len_mean_num'] = to_num(pkt_len_mean, np.nan)
        chunk['flow_iat_mean_num'] = to_num(flow_iat_mean, np.nan)


        grouped = chunk.groupby('window_id')
        buffer_rows = []
        for wid, g in grouped:
            total_pkts = int(g['fwd_pkts_num'].sum() + g['bwd_pkts_num'].sum())
            total_bytes = float(g['fwd_bytes_num'].sum() + g['bwd_bytes_num'].sum())
            # avg pkt len: weighted by packets if possible, fallback to mean
            weights = (g['fwd_pkts_num'].fillna(0) + g['bwd_pkts_num'].fillna(0))
            if weights.sum() > 0 and g['pkt_len_mean_num'].notnull().any():
                avg_pkt_len = float((g['pkt_len_mean_num'].fillna(0) * weights).sum() / weights.sum())
            else:
                avg_pkt_len = float(g['pkt_len_mean_num'].dropna().mean()) if g['pkt_len_mean_num'].dropna().size>0 else 0.0

            # mean interarrival: try flow_iat_mean weighted
            fi = g['flow_iat_mean_num'].dropna()
            if fi.size > 0:
                weights_fi = (g['fwd_pkts_num'].fillna(0) + g['bwd_pkts_num'].fillna(0)).loc[fi.index]
                if weights_fi.sum() > 0:
                    mean_interarrival = float((fi * weights_fi).sum() / weights_fi.sum())
                else:
                    mean_interarrival = float(fi.mean())
            else:
                mean_interarrival = float(np.nan)

            # unique_src/dst: try detect IP columns if present, else 0
            possible_src = [c for c in g.columns if any(x in c.lower() for x in ['src','source']) and 'ip' in c.lower()]
            possible_dst = [c for c in g.columns if any(x in c.lower() for x in ['dst','destination']) and 'ip' in c.lower()]
            unique_src = int(g[possible_src[0]].nunique()) if possible_src else 0
            unique_dst = int(g[possible_dst[0]].nunique()) if possible_dst else 0

            rssi_mean = float(np.nan)
            rssi_std = float(np.nan)
            start_time = float(g['time_parsed'].min())

            row = {
                'pcap_file': fname,
                'window_id': int(wid),
                'start_time': start_time,
                'packet_count': total_pkts,
                'total_bytes': total_bytes,
                'avg_pkt_len': avg_pkt_len,
                'unique_src': unique_src,
                'unique_dst': unique_dst,
                'mean_interarrival': mean_interarrival,
                'rssi_mean': rssi_mean,
                'rssi_std': rssi_std
            }
            buffer_rows.append(row)

        if buffer_rows:
            with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for r in buffer_rows:
                    w.writerow([r[h] for h in header])
            rows_written += len(buffer_rows)

        print(f"  chunk #{chunk_idx}: read={len(chunk)} windows={len(buffer_rows)}")

    print(f"Finished {fname}: feature rows written={rows_written}")

def convert_csv_to_parquet():
    try:
        df = pd.read_csv(OUT_CSV)
        df.to_parquet(OUT_PARQUET, index=False)
        print("Saved parquet:", OUT_PARQUET)
    except Exception as e:
        print("Parquet conversion failed:", e)

def main():
    files = sorted(glob.glob(os.path.join(CIC_CSV_DIR, "*.csv")))
    if not files:
        print("No cic csv files found in", CIC_CSV_DIR)
        return
    for f in files:
        try:
            process_file(f)
        except Exception as e:
            print("Error processing", f, "->", e)
    convert_csv_to_parquet()
    print("All done. Output CSV:", OUT_CSV)

if __name__ == "__main__":
    main()
