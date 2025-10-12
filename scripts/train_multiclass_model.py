# scripts/train_multiclass_model.py
"""
Train a single multiclass model (phishing / adult / legitimate).

Outputs:
 - model_service/models/pretrained/multiclass_xgb.json  (xgboost model)
 - model_service/models/pretrained/tfidf_vectorizer.joblib
 - model_service/models/pretrained/numeric_features.joblib   (list of numeric feature names)
 - model_service/models/pretrained/label_map.joblib
"""

import os
from pathlib import Path
import re
import math
import joblib
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import tldextract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
from scipy import sparse

# ---------- Paths ----------
DATA_PATH = "data/processed/url_features.csv"   # change if different
OUT_DIR = Path("model_service/models/pretrained")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUT = OUT_DIR / "multiclass_xgb.json"
VECT_OUT = OUT_DIR / "tfidf_vectorizer.joblib"
NUM_FEAT_OUT = OUT_DIR / "numeric_features.joblib"
LABEL_MAP_OUT = OUT_DIR / "label_map.joblib"

# ---------- Utilities ----------
def entropy(s):
    if not s:
        return 0.0
    from collections import Counter
    probs = [v/len(s) for v in Counter(s).values()]
    return -sum(p*math.log2(p) for p in probs if p > 0)

def extract_url_fields(url: str):
    u = str(url).strip()
    if not u:
        u = ""
    if not u.startswith(("http://", "https://")):
        u2 = "http://" + u
    else:
        u2 = u
    parsed = urlparse(u2)
    domain_full = parsed.netloc.lower()
    tx = tldextract.extract(u2)
    domain = tx.domain or ""
    subdomain = tx.subdomain or ""
    path = parsed.path or ""
    query = parsed.query or ""
    host = domain_full
    domain_path = (host + " " + path + " " + query).strip()
    feats = {
        "url_length": len(u),
        "domain_length": len(domain),
        "subdomain_count": subdomain.count('.') + 1 if subdomain else 0,
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": host.count('.'),
        "num_hyphen": host.count('-'),
        "num_underscore": host.count('_'),
        "num_slash": path.count('/'),
        "num_digits": sum(1 for c in u if c.isdigit()),
        "digit_ratio": sum(1 for c in u if c.isdigit()) / max(1, len(u)),
        "entropy_domain": entropy(domain),
        "has_https": int(u.lower().startswith("https")),
        "has_ip": int(bool(re.match(r'^\d+\.\d+\.\d+\.\d+$', tx.domain))),
        "has_porn_token": int(any(k in u.lower() for k in ("porn", "xxx", "sex", "adult", "cam", "tube"))),
    }
    return feats, domain_path

# ---------- Load data ----------
print("Loading:", DATA_PATH)
df = pd.read_csv(DATA_PATH, on_bad_lines='skip', quoting=3)
df.columns = df.columns.str.strip()
if 'type' in df.columns and 'label' not in df.columns:
    df = df.rename(columns={'type': 'label'})
if 'label' not in df.columns:
    raise SystemExit("Dataset must contain 'label' column with values: phishing/adult/legitimate")

df['label'] = df['label'].astype(str).str.strip().str.lower()

allowed = {'phishing', 'adult', 'legitimate', 'benign'}
df['label'] = df['label'].apply(lambda x: x if x in allowed else 'legitimate')

print("Total rows:", len(df))

# ---------- Feature extraction ----------
texts = []
numeric_list = []
for u in df['url'].astype(str).fillna("").tolist():
    feats, domain_path = extract_url_fields(u)
    numeric_list.append(feats)
    texts.append(domain_path)

num_df = pd.DataFrame(numeric_list).fillna(0)

# ---------- Optional models ----------
extra_feat_names = []
def try_add_xgb_score(path, name, X_numeric_for_xgb):
    if not Path(path).exists():
        print(f"Optional model {path} not found, skipping {name}")
        return None
    try:
        model_obj = joblib.load(path)
        print(f"Loaded optional XGBoost model: {path}")
        d = xgb.DMatrix(X_numeric_for_xgb)
        probs = model_obj.predict(d)
        return probs
    except Exception as e:
        print("Failed to use optional xgb model:", e)
        return None

numeric_for_optional = num_df.copy()
phish_score = try_add_xgb_score("model_service/models/pretrained/phishing_xgb.pkl", "phishing_xgb", numeric_for_optional)
if phish_score is not None:
    num_df['phishing_xgb_score'] = phish_score
    extra_feat_names.append('phishing_xgb_score')

adult_score = try_add_xgb_score("model_service/models/pretrained/adult_xgb.pkl", "adult_xgb", numeric_for_optional)
if adult_score is not None:
    num_df['adult_xgb_score'] = adult_score
    extra_feat_names.append('adult_xgb_score')

# ---------- TF-IDF ----------
print("Fitting TF-IDF (char n-grams) on domain+path...")
tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3,6), max_features=5000)
X_text = tfidf.fit_transform(texts)
numeric_cols = list(num_df.columns)
print("Numeric feature columns:", numeric_cols)
X_num = sparse.csr_matrix(num_df.values.astype(np.float32))
X = sparse.hstack([X_num, X_text], format='csr')

# ---------- Labels ----------
label_map = {"phishing": 0, "adult": 1, "legitimate": 2}
y = df['label'].map(label_map).values

if np.isnan(y).any():
    missing_count = np.isnan(y).sum()
    print(f"⚠️ Found {missing_count} missing target labels. Dropping those rows.")
    notnull_mask = ~np.isnan(y)
    X = X[notnull_mask]
    y = y[notnull_mask]

# ---------- Train / eval split ----------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train shape:", X_train.shape, "Val shape:", X_val.shape)

# ---------- Train XGBoost ----------
num_class = len(label_map)
params = {
    "objective": "multi:softprob",
    "num_class": num_class,
    "eval_metric": "mlogloss",
    "eta": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbosity": 1
}

# Convert string labels to integers safely
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)

# Save label encoder + map
joblib.dump(label_encoder, OUT_DIR / "label_encoder.joblib")
label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
joblib.dump(label_map, LABEL_MAP_OUT)

# Train with early stopping
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
watchlist = [(dtrain, "train"), (dval, "val")]

print("Training XGBoost multiclass...")
num_round = 300
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=num_round,
    evals=watchlist,
    early_stopping_rounds=30
)

# ---------- Save all artifacts ----------
bst.save_model(str(MODEL_OUT))
joblib.dump(tfidf, VECT_OUT)
joblib.dump(numeric_cols, NUM_FEAT_OUT)
print("✅ Saved model to:", MODEL_OUT)
print("✅ Saved vectorizer to:", VECT_OUT)
print("✅ Saved numeric feature list to:", NUM_FEAT_OUT)
print("✅ Saved label map to:", LABEL_MAP_OUT)

# ---------- Evaluate ----------
dval_full = xgb.DMatrix(X_val)
preds = bst.predict(dval_full)
y_pred = np.argmax(preds, axis=1)

print("\nClassification report:")
target_names = [str(c) for c in label_encoder.classes_]
print(classification_report(y_val, y_pred, target_names=target_names))
print("Confusion matrix:")
print(confusion_matrix(y_val, y_pred))
print(f"\n✅ Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
