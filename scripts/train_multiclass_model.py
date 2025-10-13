# scripts/train_multiclass_model.py
"""
Train a single multiclass model (phishing / adult / legitimate).

Outputs:
 - model_service/models/pretrained/multiclass_xgb.json  (xgboost model)
 - model_service/models/pretrained/tfidf_vectorizer.joblib  (word + char tuple)
 - model_service/models/pretrained/numeric_features.joblib
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
DATA_PATH = "data/processed/url_features.csv"
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

    # ✅ Extended adult and phishing keyword sets
    adult_keywords = ("porn", "xxx", "sex", "adult", "cam", "tube", "nude", "hot", "fuck", "escort", "babe", "boobs")
    phish_keywords = ("login", "signin", "verify", "update", "account", "secure", "bank", "confirm", "wallet", "reset")

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

        # ✅ New adult + phishing features
        "has_adult_keyword": int(any(k in u.lower() for k in adult_keywords)),
        "has_phish_keyword": int(any(k in u.lower() for k in phish_keywords)),

        # ✅ Extra engineered features
        "count_special_chars": sum(c in u for c in "@%=&?~"),
        "is_shortened_url": int(any(x in u.lower() for x in ("bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly"))),
        "token_count_path": path.count('/') + 1 if path else 0,
        "has_login_token": int(any(k in u.lower() for k in ("login", "signin", "verify", "update", "account", "secure"))),
        "tld_type": 0 if host.endswith((".gov", ".edu", ".org")) else 1
    }

    return feats, domain_path


# ---------- Load Data ----------
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

# ---------- Feature Extraction ----------
texts = []
numeric_list = []
for u in df['url'].astype(str).fillna("").tolist():
    feats, domain_path = extract_url_fields(u)
    numeric_list.append(feats)
    texts.append(domain_path)

num_df = pd.DataFrame(numeric_list).fillna(0)

# ---------- TF-IDF ----------
print("Fitting TF-IDF (word + char n-grams)...")
from scipy.sparse import hstack
tfidf_word = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=3000)
tfidf_char = TfidfVectorizer(analyzer='char', ngram_range=(3,5), max_features=3000)

X_word = tfidf_word.fit_transform(texts)
X_char = tfidf_char.fit_transform(texts)
X_text = hstack([X_word, X_char])

# ✅ Save both vectorizers together as tuple
joblib.dump((tfidf_word, tfidf_char), VECT_OUT)

numeric_cols = list(num_df.columns)
print("Numeric feature columns:", numeric_cols)
X_num = sparse.csr_matrix(num_df.values.astype(np.float32))
X = sparse.hstack([X_num, X_text], format='csr')

# ---------- Labels ----------
label_map = {"phishing": 0, "adult": 1, "legitimate": 2}
y = df['label'].map(label_map).values
if np.isnan(y).any():
    notnull_mask = ~np.isnan(y)
    X = X[notnull_mask]
    y = y[notnull_mask]

# ---------- Train/Test Split ----------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train shape:", X_train.shape, "Val shape:", X_val.shape)

# ---------- Train Model ----------
num_class = len(label_map)
params = {
    "objective": "multi:softprob",
    "num_class": num_class,
    "eval_metric": "mlogloss",
    "eta": 0.1,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "seed": 42,
    "verbosity": 1
}

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)
joblib.dump(label_encoder, OUT_DIR / "label_encoder.joblib")

label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
joblib.dump(label_map, LABEL_MAP_OUT)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
watchlist = [(dtrain, "train"), (dval, "val")]

print("Training XGBoost multiclass...")
bst = xgb.train(params, dtrain, num_boost_round=300, evals=watchlist, early_stopping_rounds=30)

# ---------- Save Artifacts ----------
bst.save_model(str(MODEL_OUT))
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
