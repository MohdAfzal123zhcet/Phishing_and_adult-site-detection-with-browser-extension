# scripts/evaluate_external.py
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import sparse
from scipy.sparse import hstack
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from urllib.parse import urlparse
import tldextract
import re
from collections import Counter
import math
import os

# ---------- Paths ----------
MODEL_PATH = "model_service/models/pretrained/multiclass_xgb.json"
VECT_PATH = "model_service/models/pretrained/tfidf_vectorizer.joblib"
NUM_FEAT_PATH = "model_service/models/pretrained/numeric_features.joblib"
LABEL_MAP_PATH = "model_service/models/pretrained/label_map.joblib"
DATA_PATH = "data/external/test_urls.csv"

# ---------- Load Artifacts ----------
print("🔹 Loading model and vectorizer...")
bst = xgb.Booster()
bst.load_model(MODEL_PATH)

tfidf_word, tfidf_char = joblib.load(VECT_PATH)
numeric_cols = joblib.load(NUM_FEAT_PATH)

# --- Load and normalize label map ---
label_map_raw = joblib.load(LABEL_MAP_PATH)
if all(isinstance(k, (int, float, np.integer, np.floating)) for k in label_map_raw.keys()):
    label_map = {'phishing': 0, 'adult': 1, 'legitimate': 2}
else:
    label_map = {str(k).lower().strip(): int(v) for k, v in label_map_raw.items()}
inv_label_map = {v: k for k, v in label_map.items()}

# ---------- Utility ----------
def entropy(s):
    if not s:
        return 0.0
    probs = [v / len(s) for v in Counter(s).values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)

def normalize_host(netloc: str):
    """Normalize domain: remove port and leading 'www.'"""
    host = (netloc or "").lower().split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host
def extract_url_fields(url: str):
    """Extract numeric + structural phishing indicators (same as training)."""
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

    # ✅ Adult keywords
    adult_keywords = ("porn", "xxx", "sex", "adult", "cam", "tube",
                      "nude", "hot", "fuck", "escort", "babe", "boobs")

    # ✅ Suspicious TLDs
    suspicious_tlds = ("xyz", "top", "club", "info", "click", "link",
                       "shop", "work", "cf", "tk", "ml", "ga")

    # ✅ Numeric + structural features
    num_digits = sum(1 for c in u2 if c.isdigit())
    digit_ratio = num_digits / max(1, len(u2))
    host_parts = host.split('.') if host else []
    suffix_parts = tx.suffix.split('.') if tx.suffix else []
    registered_parts = 1 + (len(suffix_parts) if suffix_parts else 0)
    num_subdomain_parts = max(0, len(host_parts) - registered_parts)

    # ✅ New + Base features (same as training)
    feats = {
        "url_length": len(u2),
        "domain_length": len(domain),
        "subdomain_count": subdomain.count('.') + 1 if subdomain else 0,
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": host.count('.'),
        "num_hyphen": host.count('-'),
        "num_underscore": host.count('_'),
        "num_slash": path.count('/'),
        "num_digits": num_digits,
        "digit_ratio": digit_ratio,
        "entropy_domain": entropy(domain),
        "has_https": int(u2.lower().startswith("https")),
        "has_ip": int(bool(re.match(r'^\d+\.\d+\.\d+\.\d+$', tx.domain))),
        "has_adult_keyword": int(any(k in u2.lower() for k in adult_keywords)),
        "count_special_chars": sum(c in u2 for c in "@%=&?~,"),
        "is_shortened_url": int(any(x in u2.lower() for x in
                                   ("bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly"))),
        "token_count_path": path.count('/') + 1 if path else 0,
        "tld_type": 0 if host.endswith((".gov", ".edu", ".org")) else 1,
        "num_subdomain_parts": num_subdomain_parts,
        "is_suspicious_tld": int((tx.suffix or "").lower() in suspicious_tlds),
        "many_digits": int(num_digits >= 5),
        "many_dots_or_hyphens": int(host.count('.') >= 4 or host.count('-') >= 4),
        "deep_subdomain": int(num_subdomain_parts >= 3),
        "long_url_flag": int(len(u2) > 120 and len(query) > 30),
        "no_https_with_digits": int(not u2.lower().startswith("https") and num_digits >= 3),

        # 🔹 Newly added phishing indicators (match training)
        "num_parameters": len(query.split("&")) if query else 0,
        "has_at_symbol": int("@" in u2),
        "has_comma_symbol": int("," in u2),
        "contains_equal_sign": int("=" in u2),
        "contains_hex_encoding": int(bool(re.search(r'%[0-9a-fA-F]{2}', u2))),
        "url_depth": path.count('/'),
        "contains_digit_in_domain": int(any(ch.isdigit() for ch in domain)),
        "contains_dash_in_domain": int("-" in domain),
    }

    domain_path = (host + " " + path + " " + query).strip()
    return feats, domain_path, normalize_host(parsed.netloc)

# ---------- Load External Data ----------
print("🔹 Loading external test data...")
df = pd.read_csv(DATA_PATH)
if 'url' not in df.columns or 'label' not in df.columns:
    raise ValueError("CSV must contain 'url' and 'label' columns!")

df['label'] = df['label'].astype(str).str.lower().str.strip()
valid_labels = set(label_map.keys())
df = df[df['label'].isin(valid_labels)].reset_index(drop=True)

urls = df['url'].astype(str).tolist()
labels = df['label'].map(label_map).values
print(f"✅ Loaded {len(urls)} URLs.")

# ---------- Feature Extraction ----------
num_feat_list, text_list, host_list = [], [], []
for u in urls:
    feats, text, host_norm = extract_url_fields(u)
    num_feat_list.append([feats.get(c, 0) for c in numeric_cols])
    text_list.append(text)
    host_list.append(host_norm)

X_num = sparse.csr_matrix(num_feat_list, dtype=np.float64)
X_word = tfidf_word.transform(text_list)
X_char = tfidf_char.transform(text_list)
X_text = hstack([X_word, X_char])
X = hstack([X_num, X_text], format="csr").astype(np.float64)

# ---------- Predict ----------
print("🔹 Predicting (pure model + exact whitelist)...")
xgb.set_config(verbosity=0)
np.random.seed(42)
dtest = xgb.DMatrix(X)
y_pred_proba = bst.predict(dtest)
y_pred = np.argmax(y_pred_proba, axis=1)

# ✅ Exact-domain whitelist override
whitelist_exact = {
    "youtube.com", "google.com", "facebook.com", "linkedin.com",
    "github.com", "twitter.com"
}
for i, host in enumerate(host_list):
    if host in whitelist_exact:
        y_pred[i] = label_map["legitimate"]

# ---------- Evaluate ----------
acc = accuracy_score(labels, y_pred)
print(f"\n✅ Accuracy on External Test Set (model + exact whitelist): {acc:.3f}\n")
label_order = [label_map['phishing'], label_map['adult'], label_map['legitimate']]
target_names = ['phishing', 'adult', 'legitimate']
print(classification_report(labels, y_pred, labels=label_order, target_names=target_names, digits=2, zero_division=0))
print("\nConfusion matrix:")
print(confusion_matrix(labels, y_pred, labels=label_order))

# ---------- Save Results ----------
os.makedirs("data/external", exist_ok=True)
results_df = pd.DataFrame({
    "url": urls,
    "normalized_host": host_list,
    "true_label": [inv_label_map[i] for i in labels],
    "predicted_label": [inv_label_map[i] for i in y_pred],
    "confidence": [float(np.max(p)) for p in y_pred_proba]
})
results_df.to_csv("data/external/predictions_report.csv", index=False)
print("\n📝 Saved detailed results to: data/external/predictions_report.csv")
