
# Bu script cic_features/cic_mapped.parquet dosyasını bulur, backup alır,
# mean_interarrival birimini otomatik düzeltir ve aynı dosyayı günceller.

import sys, shutil, os
import pandas as pd
import numpy as np
from pathlib import Path

candidates = [
    "cic_features/cic_mapped.parquet",
    "cic_features/cic_mapped.csv",
    "data/cic_mapped.parquet",
    "data/cic_mapped.csv",
    "cic_mapped.parquet",
    "cic_mapped.csv"
]

found = None
for p in candidates:
    if Path(p).exists():
        found = Path(p)
        break

if not found:
    print("HATA: cic_mapped dosyası bulunamadı. Aşağıdaki yolları kontrol et:")
    for c in candidates:
        print("  -", c)
    sys.exit(2)

print("Bulunan dosya:", found)

# yükle parquet veya csv
if found.suffix == ".parquet":
    df = pd.read_parquet(found)
else:
    df = pd.read_csv(found)

print("Loaded shape:", df.shape)
if 'mean_interarrival' not in df.columns:
    print("HATA: 'mean_interarrival' kolonu yok. Dosya kolonlarını gösteriyorum:")
    print(df.columns.tolist())
    sys.exit(3)

med_before = df['mean_interarrival'].median(skipna=True)
print("mean_interarrival median before:", med_before)

# otomatik düzeltme mantığı
converted = False
if pd.isna(med_before):
    print("median NaN — hiçbir dönüşüm uygulanamayacak.")
else:
    if med_before > 1e4:
        # çok büyük -> mikro saniye veya ns muhtemel; önce mikrosec varsayıp /1e6 dene
        print("Muhtemel mikro-saniye birimi tespit edildi -> /1e6 uygulanıyor (micro->s)")
        df['mean_interarrival'] = df['mean_interarrival'] / 1e6
        converted = True
    elif med_before > 1e1 and med_before < 1e4:
        print("Muhtemel milisaniye birimi tespit edildi -> /1e3 uygulanıyor (ms->s)")
        df['mean_interarrival'] = df['mean_interarrival'] / 1e3
        converted = True
    else:
        print("No unit conversion applied (median küçük).")

# clip uç değerleri (0..99.9 perc)
low_clip = 0.0
high_clip = df['mean_interarrival'].quantile(0.999)
df['mean_interarrival'] = df['mean_interarrival'].clip(lower=low_clip, upper=high_clip)

med_after = df['mean_interarrival'].median(skipna=True)
print("mean_interarrival median after:", med_after)

# rssi doldurma uyarısı: eğer tüm rssi NaN ise bilgi ver
if 'rssi_mean' in df.columns:
    n_rssi = df['rssi_mean'].isna().sum()
    total = len(df)
    print(f"rssi_mean NaN: {n_rssi}/{total}")
else:
    print("rssi_mean kolonu yok.")

# backup orijinali
backup = found.with_suffix(found.suffix + ".orig")
if not backup.exists():
    print("Orijinal dosya yedekleniyor ->", backup)
    shutil.copy(found, backup)
else:
    print("Backup zaten mevcut:", backup)

# overwrite same path (parquet veya csv)
if found.suffix == ".parquet":
    df.to_parquet(found, index=False)
else:
    df.to_csv(found, index=False)

print("Güncelleme tamamlandı. Kaydedildi:", found)
print("Not: Eğer cross_test_cic.py başka bir dosya bekliyorsa, script'i o dosya adına göre çalıştır.")
