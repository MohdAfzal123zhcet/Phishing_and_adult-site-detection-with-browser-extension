# model_service/trainers/train_xgb.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import joblib

# -------------------------
DATA_PATH = "data/processed/url_features.csv"
MODEL_PATH = "model_service/models/pretrained/phishing_xgb.pkl"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# -------------------------
print("Loading dataset:", DATA_PATH)
# Auto detect separator (tab or comma)
try:
    df = pd.read_csv(DATA_PATH, sep=None, engine="python", on_bad_lines='skip', quoting=3)
except Exception as e:
    print("Auto-detect failed, retrying with tab separator...")
    df = pd.read_csv(DATA_PATH, sep="\t", on_bad_lines='skip', quoting=3)

# Clean column names
df.columns = df.columns.str.strip()
print("Columns found:", list(df.columns))

# -------------------------
# If 'type' column exists, rename it to 'label'
if "label" not in df.columns and "type" in df.columns:
    df.rename(columns={"type": "label"}, inplace=True)

if "label" not in df.columns:
    raise ValueError("Dataset must have 'label' or 'type' column. Found:", df.columns)

# Normalize label text
df["label"] = df["label"].astype(str).str.strip().str.lower()

# Binary target: phishing=1, else 0
df["is_phish"] = df["label"].apply(lambda x: 1 if x in ["phishing", "malicious", "phish"] else 0)

# -------------------------
# Ensure 'url' column exists
if "url" not in df.columns:
    raise ValueError("Dataset must contain a 'url' column")

# -------------------------
# Generate simple numeric features if not present
from urllib.parse import urlparse
import re

def extract_basic(url):
    u = str(url).strip()
    parsed = urlparse(u if "://" in u else "http://" + u)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    query = parsed.query.lower()
    return {
        "url_length": len(u),
        "domain_length": len(domain),
        "path_length": len(path),
        "num_dots": domain.count(".") + path.count("."),
        "num_digits": len(re.findall(r"\d", u)),
        "has_https": int("https" in u.lower()),
        "has_porn": int(any(k in u.lower() for k in ["porn", "sex", "xxx", "adult"]))
    }

print("Extracting basic URL features...")
basic_feats = df["url"].astype(str).apply(extract_basic).apply(pd.Series)
df = pd.concat([df.reset_index(drop=True), basic_feats.reset_index(drop=True)], axis=1)

# -------------------------
# Prepare train data
drop_cols = ["url", "label"]
feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype in (np.int64, np.float64)]
X = df[feature_cols].fillna(0)
y = df["is_phish"].astype(int)

print(f"✅ Samples: {len(X)} | Features: {len(feature_cols)}")

if len(X) < 10:
    raise ValueError("Not enough samples to train the model!")

# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# -------------------------
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 5,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}

print("Training XGBoost model...")
bst = xgb.train(params, dtrain, num_boost_round=150, evals=[(dtest, "test")], early_stopping_rounds=10, verbose_eval=20)

# -------------------------
y_prob = bst.predict(dtest)
y_pred = (y_prob > 0.5).astype(int)

print("\n✅ Evaluation Results:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("AUC:", round(roc_auc_score(y_test, y_prob), 4))
print(classification_report(y_test, y_pred, target_names=["legitimate", "phishing"]))

# -------------------------
joblib.dump(bst, MODEL_PATH)
print(f"\n💾 Model saved successfully → {MODEL_PATH}")
