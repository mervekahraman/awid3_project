import joblib, os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from config import FEATURE_DIR, MODEL_DIR
from sklearn.model_selection import train_test_split

scaler_pkg = joblib.load(os.path.join(MODEL_DIR, "scaler_awid.joblib"))
scaler = scaler_pkg['scaler']
features = scaler_pkg['features']

df = pd.read_parquet(os.path.join(FEATURE_DIR, "features_clean.parquet"))
# train on normal windows ONLY
from config import NORMAL_PCAPS
train_df = df[df['pcap_file'].isin(NORMAL_PCAPS)]
X_train = scaler.transform(train_df[features].values)

os.makedirs(MODEL_DIR, exist_ok=True)

# IsolationForest
iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42, n_jobs=-1)
iso.fit(X_train)
joblib.dump({'model':iso, 'scaler':scaler, 'features':features}, os.path.join(MODEL_DIR, "iso_model.joblib"))
print("Saved IsolationForest.")

# One-Class SVM
ocsvm = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
ocsvm.fit(X_train)
joblib.dump({'model':ocsvm, 'scaler':scaler, 'features':features}, os.path.join(MODEL_DIR, "ocsvm_model.joblib"))
print("Saved One-Class SVM.")
