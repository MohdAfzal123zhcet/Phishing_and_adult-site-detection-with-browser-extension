# scripts/train_multiclass_model.py
"""
Train a single multiclass model (phishing / adult / legitimate)
with structural phishing detection (no phishing keywords).
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
    probs = [v / len(s) for v in Counter(s).values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def extract_url_fields(url: str):
    u = str(url).strip()
    if not u:
        u = ""
    if not u.startswith(("http://", "https://")):
        u2 = "http://" + u
    else:
        u2 = u

    u2 = re.sub(r'[\[\]#\s]+', '', u2)
    if re.match(r'^(https?://)\d', u2):
        u2 = re.sub(r'^(https?://)(\d)', r'\1x-\2', u2)

    try:
        parsed = urlparse(u2)
    except ValueError:
        cleaned = re.sub(r'[^A-Za-z0-9\-._:/\?\&=#%]', '', u2)
        parsed = urlparse(cleaned)

    domain_full = parsed.netloc.lower()
    tx = tldextract.extract(u2)
    domain = tx.domain or ""
    subdomain = tx.subdomain or ""
    path = parsed.path or ""
    query = parsed.query or ""
    host = domain_full

    # ✅ Adult keyword set (as before)
    adult_keywords = ("porn", "xxx", "sex", "adult", "cam", "tube", "nude", "hot", "fuck", "escort", "babe", "boobs")

    # ✅ Suspicious TLDs
    suspicious_tlds = ("xyz", "top", "club", "info", "click", "link", "shop", "work", "cf", "tk", "ml", "ga")

    # ✅ Numeric and structural base
    num_digits = sum(1 for c in u2 if c.isdigit())
    num_dots = host.count('.')
    num_hyphen = host.count('-')
    url_len = len(u2)
    query_len = len(query)
    has_https = int(u2.lower().startswith("https"))

    # ✅ Domain decomposition
    host_parts = host.split('.') if host else []
    suffix_parts = tx.suffix.split('.') if tx.suffix else []
    registered_parts = 1 + (len(suffix_parts) if suffix_parts else 0)
    num_subdomain_parts = max(0, len(host_parts) - registered_parts)

    # ✅ Extra URL-based phishing indicators (NEW)
    feats = {
        "url_length": url_len,
        "domain_length": len(domain),
        "subdomain_count": subdomain.count('.') + 1 if subdomain else 0,
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": num_dots,
        "num_hyphen": num_hyphen,
        "num_underscore": host.count('_'),
        "num_slash": path.count('/'),
        "num_digits": num_digits,
        "digit_ratio": num_digits / max(1, len(u2)),
        "entropy_domain": entropy(domain),
        "has_https": has_https,
        "has_ip": int(bool(re.match(r'^\d+\.\d+\.\d+\.\d+$', tx.domain))),
        "has_adult_keyword": int(any(k in u2.lower() for k in adult_keywords)),
        "count_special_chars": sum(c in u2 for c in "@%=&?~,"),
        "is_shortened_url": int(any(x in u2.lower() for x in ("bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly"))),
        "token_count_path": path.count('/') + 1 if path else 0,
        "tld_type": 0 if host.endswith((".gov", ".edu", ".org")) else 1,

        # 🔹 Newly Added Features
        "num_parameters": len(query.split("&")) if query else 0,
        "has_at_symbol": int("@" in u2),
        "has_comma_symbol": int("," in u2),
        "contains_equal_sign": int("=" in u2),
        "contains_hex_encoding": int(bool(re.search(r'%[0-9a-fA-F]{2}', u2))),
        "url_depth": path.count('/'),
        "contains_digit_in_domain": int(any(ch.isdigit() for ch in domain)),
        "contains_dash_in_domain": int("-" in domain),
    }

    # ✅ Advanced phishing indicators (same as before)
    tld = (tx.suffix or "").lower()
    feats["is_suspicious_tld"] = int(tld in suspicious_tlds)
    feats["many_digits"] = int(num_digits >= 5)
    feats["many_dots_or_hyphens"] = int(num_dots >= 4 or num_hyphen >= 4)
    feats["deep_subdomain"] = int(num_subdomain_parts >= 3)
    feats["long_url_flag"] = int(url_len > 120 and query_len > 30)
    feats["no_https_with_digits"] = int(has_https == 0 and num_digits >= 3)

    domain_path = (host + " " + path + " " + query).strip()
    return feats, domain_path


# ---------- Load Data ----------
print("Loading:", DATA_PATH)
df = pd.read_csv(DATA_PATH, on_bad_lines='skip', quoting=3)
df.columns = df.columns.str.strip()

if 'type' in df.columns and 'label' not in df.columns:
    df = df.rename(columns={'type': 'label'})
if 'label' not in df.columns:
    raise SystemExit("Dataset must contain 'label' column")

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
tfidf_word = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), max_features=8000)
tfidf_char = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=8000)

X_word = tfidf_word.fit_transform(texts)
X_char = tfidf_char.fit_transform(texts)
X_text = sparse.hstack([X_word, X_char])

joblib.dump((tfidf_word, tfidf_char), VECT_OUT)

numeric_cols = list(num_df.columns)
print("Numeric feature columns:", numeric_cols)
X_num = sparse.csr_matrix(num_df.values.astype(np.float32))
X = sparse.hstack([X_num, X_text], format='csr')

# ---------- Labels ----------
label_map = {"phishing": 0, "adult": 1, "legitimate": 2}

# 🩵 FIX: Keep legitimate class and merge benign into legitimate
df['label'] = df['label'].replace('benign', 'legitimate')
df = df[df['label'].isin(label_map.keys())]
print("Label counts:\n", df['label'].value_counts())

y = df['label'].map(label_map).values
X = X[:len(y)]  # ensure same shape alignment

# ---------- Train/Test Split ----------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train shape:", X_train.shape, "Val shape:", X_val.shape)

# ---------- Model Parameters ----------
num_class = len(label_map)
params = {
    "objective": "multi:softprob",
    "num_class": num_class,
    "eval_metric": "mlogloss",
    "eta": 0.07,
    "max_depth": 7,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "scale_pos_weight": 1.5,  # ⚡ Boost phishing detection
    "seed": 42,
    "verbosity": 1,
}

# ---------- Train Model ----------
print("Training XGBoost multiclass...")
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
watchlist = [(dtrain, "train"), (dval, "val")]

bst = xgb.train(params, dtrain, num_boost_round=400, evals=watchlist, early_stopping_rounds=40)

# ---------- Save Artifacts ----------
bst.save_model(str(MODEL_OUT))
joblib.dump(numeric_cols, NUM_FEAT_OUT)
joblib.dump(label_map, LABEL_MAP_OUT)

print("✅ Model, vectorizer & features saved successfully!")

# ---------- Evaluate ----------
dval_full = xgb.DMatrix(X_val)
preds = bst.predict(dval_full)
y_pred = np.argmax(preds, axis=1)

print("\nClassification report:")
print(classification_report(y_val, y_pred, target_names=["phishing", "adult", "legitimate"]))
print("Confusion matrix:")
print(confusion_matrix(y_val, y_pred))
print(f"\n✅ Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
