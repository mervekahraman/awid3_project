import argparse
from pcap_to_csv import convert_all_pcaps
from feature_extraction import extract_all_features
from train_eval import train_isolation_forest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default="all",
                        choices=["pcap2csv","features","train","all"],
                        help="Hangi adımı çalıştırmak istersin?")
    args = parser.parse_args()

    if args.step in ("pcap2csv","all"):
        print("PCAP -> CSV dönüştürülüyor...")
        convert_all_pcaps()
    if args.step in ("features","all"):
        print("Özellikler çıkarılıyor...")
        extract_all_features()
    if args.step in ("train","all"):
        print("Model eğitimi ve değerlendirme başlıyor...")
        train_isolation_forest()

if __name__ == "__main__":
    main()
