
# Robust, streaming CSV -> windowed features using csv.reader with escapechar='\\'
import csv
import os
import glob
import re
from pathlib import Path
import numpy as np
import pandas as pd
from config import WINDOW_SEC, CSV_CHUNKSIZE
from utils import ensure_dirs

ensure_dirs(["features"])

OUT_CSV = os.path.join("features", "features.csv")
OUT_PARQUET = os.path.join("features", "features.parquet")

# header write/overwrite
def init_out_csv():
    header = ['pcap_file','window_id','start_time','packet_count','total_bytes','avg_pkt_len',
              'unique_src','unique_dst','mean_interarrival','rssi_mean','rssi_std']
    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

def append_rows_to_csv(rows):
    if not rows:
        return
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow([
                r['pcap_file'], r['window_id'], r['start_time'],
                r['packet_count'], r['total_bytes'], r['avg_pkt_len'],
                r['unique_src'], r['unique_dst'], r['mean_interarrival'],
                r['rssi_mean'], r['rssi_std']
            ])

# clean time bytes/str -> keep only allowed chars
_ALLOWED_TIME_RE = re.compile(rb'[^0-9\.\-eE\+]')

def clean_time_bytes(b: bytes) -> bytes:
    # remove backslashes, invisible bytes if any, then allow only ascii digits/dot/minus/e/E/+
    if b is None:
        return b''
    # replace backslash bytes
    b = b.replace(b'\\', b'')
    # remove BOM-like bytes (utf-8 BOM)
    if b.startswith(b'\xef\xbb\xbf'):
        b = b[3:]
    # remove non-allowed bytes
    b = _ALLOWED_TIME_RE.sub(b'', b)
    return b

# parse rssi string like "-34,-40,-34" or "-34\,-40\,-34" into mean (float)
def parse_rssi_field(s: str):
    if s is None:
        return np.nan
    try:
        s2 = s.replace("\\", "")
        parts = [p.strip() for p in s2.split(",") if p.strip()!='']
        vals = []
        for p in parts:
            try:
                vals.append(float(p))
            except:
                pass
        return float(np.mean(vals)) if vals else np.nan
    except Exception:
        return np.nan

def process_single_file_streaming(csv_path: str):
    """
    Stream parse a csv_exports/*.csv file using csv.reader with escapechar='\\'
    Build windows online (agg per window) to avoid storing per-packet rows.
    Returns number of feature rows written for this file.
    """
    fname = os.path.basename(csv_path)
    rows_written = 0

    cur_win = None
    cur_tmin = None
    # aggregator structure for current window
    agg = None

    # open file in binary mode and use text wrapper for csv.reader to set escapechar
    with open(csv_path, "rb") as bf:
        # Read first chunk to detect newline encoding and header

        bf.seek(0)
        # Use a text wrapper
        import io
        text_stream = io.TextIOWrapper(bf, encoding='utf-8', errors='surrogateescape', newline='')
        reader = csv.reader(text_stream, delimiter=',', escapechar='\\')
        # read header
        try:
            header = next(reader)
        except StopIteration:
            print(f"File empty: {csv_path}")
            return 0
        # find column indices (header contains expected names)

        cols = [c.strip() for c in header]
        # get indices, default to known positions if exact names absent
        def idx_of(possible):
            for p in possible:
                if p in cols:
                    return cols.index(p)
            return None

        time_idx = idx_of(['frame.time_epoch','frame.time','frame.time_relative','epoch_time','time_epoch'])
        len_idx = idx_of(['frame.len'])
        wlan_sa_idx = idx_of(['wlan.sa'])
        wlan_da_idx = idx_of(['wlan.da'])
        eth_src_idx = idx_of(['eth.src'])
        eth_dst_idx = idx_of(['eth.dst'])
        rssi_idx = idx_of(['radiotap.dbm_antsignal'])

        # Fallback index positions if header names missing (best-effort)
        # Typically header order is: frame.time_epoch,frame.len,eth.src,eth.dst,wlan.sa,wlan.da,radiotap.dbm_antsignal
        if time_idx is None:
            time_idx = 0
        if len_idx is None:
            len_idx = 1

        # iterate rows
        line_no = 1  # header consumed
        for row in reader:
            line_no += 1
            # make sure row has at least up to max index; if not, pad
            if len(row) <= max(time_idx, len_idx):
                # malformed row — skip
                continue

            # raw time bytes: we need the original bytes, but csv.reader returned text decoded via surrogateescape,
            # so to get original bytes we can encode with 'utf-8' + 'surrogateescape' to recover bytes
            try:
                raw_time_text = row[time_idx]
                # recover original bytes
                raw_time_bytes = raw_time_text.encode('utf-8', 'surrogateescape')
                cleaned = clean_time_bytes(raw_time_bytes)
                if not cleaned:
                    # skip if empty after cleaning
                    continue
                # decode cleaned as ascii-safe string
                time_str = cleaned.decode('ascii', errors='ignore')
                # convert to float
                try:
                    tval = float(time_str)
                except:
                    # skip unparsable
                    continue
            except Exception:
                continue

            # len value
            length_val = None
            try:
                raw_len = row[len_idx] if len_idx < len(row) else ''
                # remove any non-digit/dot/minus
                s_len = re.sub(r'[^0-9\.\-]', '', raw_len or '')
                length_val = float(s_len) if s_len != '' else 0.0
            except:
                length_val = 0.0

            # rssi parse
            rssi_val = np.nan
            if rssi_idx is not None and rssi_idx < len(row):
                rssi_val = parse_rssi_field(row[rssi_idx])

            # src/dst
            src = None
            dst = None
            if wlan_sa_idx is not None and wlan_sa_idx < len(row):
                src = row[wlan_sa_idx].strip() or None
            elif eth_src_idx is not None and eth_src_idx < len(row):
                src = row[eth_src_idx].strip() or None
            if wlan_da_idx is not None and wlan_da_idx < len(row):
                dst = row[wlan_da_idx].strip() or None
            elif eth_dst_idx is not None and eth_dst_idx < len(row):
                dst = row[eth_dst_idx].strip() or None

            # initialize tmin and current window
            if cur_tmin is None:
                cur_tmin = tval
                cur_win = int((tval - cur_tmin) // WINDOW_SEC)

            win_id = int((tval - cur_tmin) // WINDOW_SEC)
            # if same window, aggregate; if moved to new window, flush previous and start new
            if agg is None:
                # initialize aggregator
                agg = {
                    'pcap_file': Path(csv_path).name,
                    'window_id': win_id,
                    'start_time': float(cur_tmin + win_id * WINDOW_SEC),
                    'packet_count': 0,
                    'total_bytes': 0.0,
                    'sum_pkt_len': 0.0,
                    'unique_src_set': set(),
                    'unique_dst_set': set(),
                    'times': [],  # keep last few times to compute interarrival mean cheaply
                    'rssi_vals': []
                }

            if win_id != agg['window_id']:
                # flush current agg
                # compute avg_pkt_len and mean_interarrival
                pkt_count = agg['packet_count']
                avg_pkt = (agg['sum_pkt_len']/pkt_count) if pkt_count>0 else 0.0
                times_arr = np.array(agg['times'])
                inter = np.diff(np.sort(times_arr)) if times_arr.size>1 else np.array([])
                mean_inter = float(np.mean(inter)) if inter.size>0 else float('nan')
                rssi_mean = float(np.mean(agg['rssi_vals'])) if len(agg['rssi_vals'])>0 else float('nan')
                rssi_std = float(np.std(agg['rssi_vals'], ddof=0)) if len(agg['rssi_vals'])>0 else float('nan')
                out_row = {
                    'pcap_file': agg['pcap_file'],
                    'window_id': agg['window_id'],
                    'start_time': agg['start_time'],
                    'packet_count': pkt_count,
                    'total_bytes': float(agg['total_bytes']),
                    'avg_pkt_len': float(avg_pkt),
                    'unique_src': len(agg['unique_src_set']),
                    'unique_dst': len(agg['unique_dst_set']),
                    'mean_interarrival': mean_inter,
                    'rssi_mean': rssi_mean,
                    'rssi_std': rssi_std
                }
                append_rows_to_csv([out_row])
                rows_written += 1
                # start new aggregator for the new window
                agg = {
                    'pcap_file': Path(csv_path).name,
                    'window_id': win_id,
                    'start_time': float(cur_tmin + win_id * WINDOW_SEC),
                    'packet_count': 0,
                    'total_bytes': 0.0,
                    'sum_pkt_len': 0.0,
                    'unique_src_set': set(),
                    'unique_dst_set': set(),
                    'times': [],
                    'rssi_vals': []
                }

            # update aggregator with current packet
            agg['packet_count'] += 1
            agg['total_bytes'] += length_val
            agg['sum_pkt_len'] += length_val
            if src:
                agg['unique_src_set'].add(src)
            if dst:
                agg['unique_dst_set'].add(dst)
            agg['times'].append(tval)
            if not np.isnan(rssi_val):
                agg['rssi_vals'].append(rssi_val)

    # after file consumed, flush last aggregator
    if agg is not None and agg['packet_count']>0:
        pkt_count = agg['packet_count']
        avg_pkt = (agg['sum_pkt_len']/pkt_count) if pkt_count>0 else 0.0
        times_arr = np.array(agg['times'])
        inter = np.diff(np.sort(times_arr)) if times_arr.size>1 else np.array([])
        mean_inter = float(np.mean(inter)) if inter.size>0 else float('nan')
        rssi_mean = float(np.mean(agg['rssi_vals'])) if len(agg['rssi_vals'])>0 else float('nan')
        rssi_std = float(np.std(agg['rssi_vals'], ddof=0)) if len(agg['rssi_vals'])>0 else float('nan')
        out_row = {
            'pcap_file': agg['pcap_file'],
            'window_id': agg['window_id'],
            'start_time': agg['start_time'],
            'packet_count': pkt_count,
            'total_bytes': float(agg['total_bytes']),
            'avg_pkt_len': float(avg_pkt),
            'unique_src': len(agg['unique_src_set']),
            'unique_dst': len(agg['unique_dst_set']),
            'mean_interarrival': mean_inter,
            'rssi_mean': rssi_mean,
            'rssi_std': rssi_std
        }
        append_rows_to_csv([out_row])
        rows_written += 1

    return rows_written

def extract_all_features():
    # initialize / overwrite out csv
    init_out_csv()
    csvs = sorted(glob.glob(os.path.join("csv_exports", "*.csv")))
    total = 0
    if not csvs:
        print("No csv files in csv_exports/")
        return
    for c in csvs:
        try:
            written = process_single_file_streaming(c)
            total += written
            print(f"Wrote {written} feature rows from {os.path.basename(c)}")
        except Exception as e:
            print(f"Error processing {c}: {e}")
    print(f"TOTAL feature rows: {total}")
    # try convert to parquet
    try:
        df = pd.read_csv(OUT_CSV)
        df.to_parquet(OUT_PARQUET, index=False)
        print("Saved parquet:", OUT_PARQUET)
    except Exception as e:
        print("Parquet conversion skipped or failed:", e)
        print("CSV left at:", OUT_CSV)
