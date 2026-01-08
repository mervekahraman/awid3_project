import csv

FILE = "csv_exports/6. Kr00k.csv"

with open(FILE, "r", encoding="utf-8", errors="ignore") as f:
    sample = f.read(4096)

print("\n---- Delimiter Tespiti ----")
try:
    dialect = csv.Sniffer().sniff(sample)
    print("Detected delimiter:", repr(dialect.delimiter))
except:
    print("Delimiter tespit edilemedi, varsayılan: ','")

print("\n---- İlk Satır (Header) ----")
with open(FILE, "r", encoding="utf-8", errors="ignore") as f:
    print(f.readline().rstrip())

print("\n---- İlk 10 Satır ----")
with open(FILE, "r", encoding="utf-8", errors="ignore") as f:
    for i in range(10):
        print(f.readline().rstrip())
