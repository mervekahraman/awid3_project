#!/bin/bash
# Usage: ./score_new_pcap.sh path/to/new.pcap out_prefix

PCAP=$1
OUT_PREF=${2:-new}

TMP_CSV="tmp_${OUT_PREF}.csv"
OUT_FEATURES="features/${OUT_PREF}_features.parquet"

if [ -z "$PCAP" ]; then
  echo "Usage: $0 new.pcap [out_prefix]"
  exit 1
fi

if [ ! -f "$PCAP" ]; then
  echo "PCAP bulunamadı: $PCAP"
  exit 1
fi

mkdir -p features results

echo "1) tshark -> csv"
tshark -r "$PCAP" \
  -T fields \
  -e frame.time_epoch \
  -e frame.len \
  -e eth.src \
  -e eth.dst \
  -e wlan.sa \
  -e wlan.da \
  -e radiotap.dbm_antsignal \
  -E header=y \
  -E separator=, > "$TMP_CSV"

echo "CSV oluşturuldu: $TMP_CSV"

echo "2) Feature extraction"
python feature_extraction.py \
  --input_csv "$TMP_CSV" \
  --output_parquet "$OUT_FEATURES"

if [ ! -f "$OUT_FEATURES" ]; then
  echo "Feature extraction başarısız!"
  exit 1
fi

echo "Features üretildi: $OUT_FEATURES"

echo "3) Model scoring"
python - <<PY
import joblib, pandas as pd

scaler_pkg = joblib.load("models/scaler_awid.joblib")
scaler = scaler_pkg['scaler']
feats = scaler_pkg['features']

iso = joblib.load("models/iso_model.joblib")['model']

df = pd.read_parquet("$OUT_FEATURES")
X = df[feats].fillna(0).values
Xs = scaler.transform(X)

scores = -iso.decision_function(Xs)
df['iso_score'] = scores

out = "results/${OUT_PREF}_scores.csv"
df.to_csv(out, index=False)
print("Saved:", out)
PY

echo "Done."
