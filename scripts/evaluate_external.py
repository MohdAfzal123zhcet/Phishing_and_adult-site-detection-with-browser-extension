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


def extract_url_fields(url: str):
    """
    Extract numeric + structural phishing indicators (no keyword features).
    """
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

    # ✅ Adult keyword set (safe to keep)
    adult_keywords = ("porn", "xxx", "sex", "adult", "cam", "tube", "nude", "hot", "fuck", "escort", "babe", "boobs")

    # ✅ Numeric + structural phishing indicators only
    num_digits = sum(1 for c in u2 if c.isdigit())
    digit_ratio = num_digits / max(1, len(u2))

    host_parts = host.split('.') if host else []
    suffix_parts = tx.suffix.split('.') if tx.suffix else []
    registered_parts = 1 + (len(suffix_parts) if suffix_parts else 0)
    num_subdomain_parts = max(0, len(host_parts) - registered_parts)

    def count_digit_groups(s: str):
        return len(re.findall(r'\d+', s))
    def longest_digit_run(s: str):
        groups = re.findall(r'\d+', s)
        return max((len(g) for g in groups), default=0)
    def letters_count(s: str):
        return sum(1 for c in s if c.isalpha())

    digits_in_domain = sum(1 for c in (domain or "") if c.isdigit())
    num_digit_groups_domain = count_digit_groups(domain or "")
    longest_run_domain = longest_digit_run(domain or "")
    letters_dom = letters_count(domain or "")
    digits_vs_letters_ratio = digits_in_domain / (letters_dom + 1)

    # ✅ Suspicious TLD set
    suspicious_tlds = ("xyz", "top", "club", "info", "click", "link", "shop", "work", "cf", "tk", "ml", "ga")
    tld = (tx.suffix or "").lower()

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
        "count_special_chars": sum(c in u2 for c in "@%=&?~"),
        "is_shortened_url": int(any(x in u2.lower() for x in ("bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly"))),
        "token_count_path": path.count('/') + 1 if path else 0,
        "tld_type": 0 if host.endswith((".gov", ".edu", ".org")) else 1,

        # Advanced phishing features
        "num_subdomain_parts": num_subdomain_parts,
        "digits_in_domain": digits_in_domain,
        "num_digit_groups_domain": num_digit_groups_domain,
        "longest_run_domain": longest_run_domain,
        "digits_vs_letters_ratio": digits_vs_letters_ratio,
        "is_suspicious_tld": int(tld in suspicious_tlds),
        "many_digits": int(num_digits >= 5),
        "many_dots_or_hyphens": int(host.count('.') >= 4 or host.count('-') >= 4),
        "deep_subdomain": int(num_subdomain_parts >= 3),
        "long_url_flag": int(len(u2) > 120 and len(query) > 30),
        "no_https_with_digits": int(not u2.lower().startswith("https") and num_digits >= 3),
    }

    domain_path = (host + " " + path + " " + query).strip()
    return feats, domain_path


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
num_feat_list, text_list = [], []
for u in urls:
    feats, text = extract_url_fields(u)
    num_feat_list.append([feats.get(c, 0) for c in numeric_cols])
    text_list.append(text)

X_num = sparse.csr_matrix(num_feat_list, dtype=np.float64)
X_word = tfidf_word.transform(text_list)
X_char = tfidf_char.transform(text_list)
X_text = hstack([X_word, X_char])
X = hstack([X_num, X_text], format="csr").astype(np.float64)

# ---------- Predict ----------
print("🔹 Predicting (deterministic float64 mode)...")
xgb.set_config(verbosity=0)
np.random.seed(42)
dtest = xgb.DMatrix(X)
y_pred_proba = bst.predict(dtest)
y_pred = np.argmax(y_pred_proba, axis=1)

# ---------- Evaluate ----------
acc = accuracy_score(labels, y_pred)
print(f"\n✅ Accuracy on External Test Set: {acc:.3f}\n")
label_order = [label_map['phishing'], label_map['adult'], label_map['legitimate']]
target_names = ['phishing', 'adult', 'legitimate']
print(classification_report(labels, y_pred, labels=label_order, target_names=target_names, digits=2, zero_division=0))
print("\nConfusion matrix:")
print(confusion_matrix(labels, y_pred, labels=label_order))

# ---------- Save Results ----------
os.makedirs("data/external", exist_ok=True)
results_df = pd.DataFrame({
    "url": urls,
    "true_label": [inv_label_map[i] for i in labels],
    "predicted_label": [inv_label_map[i] for i in y_pred],
    "confidence": [float(np.max(p)) for p in y_pred_proba]
})
results_df.to_csv("data/external/predictions_report.csv", index=False)
print("\n📝 Saved detailed results to: data/external/predictions_report.csv")
