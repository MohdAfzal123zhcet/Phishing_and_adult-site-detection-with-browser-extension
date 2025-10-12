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
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from scipy import sparse

# ---------- Paths ----------
DATA_PATH = "data/processed/url_features_clean.csv"   # change if different
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
    return -sum(p*math.log2(p) for p in probs if p>0)

def extract_url_fields(url: str):
    u = str(url).strip()
    if not u:
        u = ""
    if not u.startswith(("http://","https://")):
        u2 = "http://" + u
    else:
        u2 = u
    parsed = urlparse(u2)
    domain_full = parsed.netloc.lower()
    # tldextract extracts subdomain.domain.suffix
    tx = tldextract.extract(u2)
    domain = tx.domain or ""
    subdomain = tx.subdomain or ""
    path = parsed.path or ""
    query = parsed.query or ""
    host = domain_full
    # tokens for TF-IDF: combine domain and path and query
    domain_path = (host + " " + path + " " + query).strip()
    feats = {
        "url_length": len(u),
        "domain_length": len(domain),
        "subdomain_count": subdomain.count('.')+1 if subdomain else 0,
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
        "has_porn_token": int(any(k in u.lower() for k in ("porn","xxx","sex","adult","cam","tube"))),
        # domain_path string for tfidf will be returned separately
    }
    return feats, domain_path

# ---------- Load data ----------
print("Loading:", DATA_PATH)
df = pd.read_csv(DATA_PATH, on_bad_lines='skip', quoting=3)
df.columns = df.columns.str.strip()
if 'type' in df.columns and 'label' not in df.columns:
    df = df.rename(columns={'type':'label'})
if 'label' not in df.columns:
    raise SystemExit("Dataset must contain 'label' column with values: phishing/adult/legitimate")

df['label'] = df['label'].astype(str).str.strip().str.lower()

# Keep only the 3 classes we want; if other labels exist, map them to 'legitimate'
allowed = {'phishing','adult','legitimate','benign'}
df['label'] = df['label'].apply(lambda x: x if x in allowed else 'legitimate')

print("Total rows:", len(df))
# ---------- Feature extraction ----------
texts = []
numeric_list = []
for u in df['url'].astype(str).fillna("").tolist():
    feats, domain_path = extract_url_fields(u)
    numeric_list.append(feats)
    texts.append(domain_path)

num_df = pd.DataFrame(numeric_list)
# Fill NA
num_df = num_df.fillna(0)

# ---------- Optionally add preexisting model scores as features if available ----------
# If you have preexisting phishing/adult xgb models or mlp, include their probabilities as extra features.
extra_feat_names = []
# helper to try load and predict
def try_add_xgb_score(path, name, X_numeric_for_xgb):
    if not Path(path).exists():
        print(f"Optional model {path} not found, skipping {name}")
        return None
    try:
        model_obj = joblib.load(path)
        print(f"Loaded optional XGBoost model: {path}")
        # model_obj might be Booster type
        # create DMatrix with the features the loaded model expects
        d = xgb.DMatrix(X_numeric_for_xgb)
        probs = model_obj.predict(d)
        return probs
    except Exception as e:
        print("Failed to use optional xgb model:", e)
        return None

# Prepare numeric matrix for optional models: for simple case use same numeric features
# (Optional models trained earlier might have different expected features; in that case prediction may fail.)
numeric_for_optional = num_df.copy()

# try phishing_xgb
phish_score = try_add_xgb_score("model_service/models/pretrained/phishing_xgb.pkl", "phishing_xgb", numeric_for_optional)
if phish_score is not None:
    num_df['phishing_xgb_score'] = phish_score
    extra_feat_names.append('phishing_xgb_score')

# try adult_xgb
adult_score = try_add_xgb_score("model_service/models/pretrained/adult_xgb.pkl", "adult_xgb", numeric_for_optional)
if adult_score is not None:
    num_df['adult_xgb_score'] = adult_score
    extra_feat_names.append('adult_xgb_score')

# (You can add MLP scores similarly if you have compatible code; skipping automatic MLP inclusion to avoid torch version issues.)

# ---------- TF-IDF on domain+path ----------
print("Fitting TF-IDF (char n-grams) on domain+path...")
tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3,6), max_features=5000)
X_text = tfidf.fit_transform(texts)  # sparse

# ---------- Combine numeric + text (sparse hstack) ----------
numeric_cols = list(num_df.columns)
print("Numeric feature columns:", numeric_cols)
X_num = sparse.csr_matrix(num_df.values.astype(np.float32))
X = sparse.hstack([X_num, X_text], format='csr')

# ---------- Labels (multi-class) ----------
label_map = {"phishing":0, "adult":1, "legitimate":2}
y = df['label'].map(label_map).values

# ---------- Train / eval split ----------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Train shape:", X_train.shape, "Val shape:", X_val.shape)

# ---------- Train XGBoost multiclass ----------
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

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
watchlist = [(dtrain, "train"), (dval, "val")]

print("Training XGBoost multiclass...")
bst = xgb.train(params, dtrain, num_boost_round=1000, evals=watchlist, early_stopping_rounds=20)

# Save model & artifacts
bst.save_model(str(MODEL_OUT))
joblib.dump(tfidf, VECT_OUT)
joblib.dump(numeric_cols, NUM_FEAT_OUT)
joblib.dump(label_map, LABEL_MAP_OUT)

print("Saved model to:", MODEL_OUT)
print("Saved vectorizer to:", VECT_OUT)
print("Saved numeric feature list to:", NUM_FEAT_OUT)
print("Saved label map to:", LABEL_MAP_OUT)

# ---------- Evaluate ----------
dval_full = xgb.DMatrix(X_val)
preds = bst.predict(dval_full)  # shape (n, num_class)
y_pred = np.argmax(preds, axis=1)
print("\nClassification report on validation set:")
print(classification_report(y_val, y_pred, target_names=[k for k in label_map.keys()]))

print("Confusion matrix:")
print(confusion_matrix(y_val, y_pred))
