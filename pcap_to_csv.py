import os
import shlex
from config import RAW_PCAP_DIR, CSV_DIR, TSHARK_FIELDS
from utils import run_cmd, which, ensure_dirs
from typing import List

ensure_dirs([CSV_DIR])

def pcap_to_csv(pcap_path: str, out_csv: str, fields: List[str] = None):
    """
    Tshark ile pcap -> csv
    fields: list of tshark field names
    """
    if which("tshark") is None:
        raise EnvironmentError("tshark bulunamadı. Wireshark/tshark kurulu olduğundan emin ol.")
    if fields is None:
        fields = TSHARK_FIELDS

    # -E header=y ile header eklensin
    field_args = []
    for f in fields:
        field_args += ["-e", f]
    cmd = ["tshark", "-r", pcap_path, "-T", "fields"] + field_args + ["-E", "header=y", "-E", "separator=,"]
    print("Çalıştırılıyor:", " ".join(shlex.quote(c) for c in cmd))
    res = run_cmd(cmd, check=False)
    if res.returncode != 0 and not res.stdout:
        raise RuntimeError(f"tshark hata verdi: {res.stderr}")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(res.stdout)
    print("CSV oluşturuldu:", out_csv)

def convert_all_pcaps(force=False):
    files = [f for f in os.listdir(RAW_PCAP_DIR) if f.lower().endswith(".pcap")]
    if not files:
        print("raw_pcaps içinde .pcap dosyası bulunamadı.")
        return
    for fname in files:
        in_p = os.path.join(RAW_PCAP_DIR, fname)
        base = os.path.splitext(fname)[0]
        out_p = os.path.join(CSV_DIR, base + ".csv")
        if os.path.exists(out_p) and not force:
            print("Zaten var, atlanıyor:", out_p)
            continue
        pcap_to_csv(in_p, out_p)
