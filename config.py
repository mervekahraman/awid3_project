import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_PCAP_DIR = os.path.join(ROOT, "pcaps")
CSV_DIR = os.path.join(ROOT, "csv_exports")
FEATURE_DIR = os.path.join(ROOT, "features")
MODEL_DIR = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")
CIC_CSV_DIR = os.path.join(ROOT, "cic_ids_raw")
CIC_FEATURE_DIR = os.path.join(ROOT, "cic_features")
WINDOW_SEC = 10.0
CSV_CHUNKSIZE = 200_000  # csv okurken chunk boyutu
FEATURE_OUTPUT = os.path.join(FEATURE_DIR, "features.parquet")

TSHARK_FIELDS = [
    "frame.time_epoch",
    "frame.len",
    "eth.src",
    "eth.dst",
    "wlan.sa",
    "wlan.da",
    "radiotap.dbm_antsignal"
]
NORMAL_PCAPS = [
     "1. Deauth.csv",
     "10. Malware.csv",
     "11. SSDP.csv",
     "12. Botnet.csv",
     "13. Website_spoofing.csv",
     "2. Disass.csv",
     "3.(Re)Assoc.csv",
     "4. Rogue_AP.csv",
     "5. Krack.csv",
     "6. Kr00k.csv",
     "7. Evil_Twin.csv",
     "8. SQL_Injection.csv",
     "9. SSH.csv"
]