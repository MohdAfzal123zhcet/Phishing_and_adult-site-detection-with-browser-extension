# scripts/predict_multiclass.py
import joblib
import numpy as np
import re
from urllib.parse import urlparse
import tldextract
import xgboost as xgb
from scipy import sparse
from scipy.sparse import hstack
from collections import Counter
import math

# ---------- Paths ----------
MODEL_PATH = "model_service/models/pretrained/multiclass_xgb.json"
VECT_PATH = "model_service/models/pretrained/tfidf_vectorizer.joblib"
NUM_FEAT_PATH = "model_service/models/pretrained/numeric_features.joblib"
LABEL_MAP_PATH = "model_service/models/pretrained/label_map.joblib"

# ---------- Load Artifacts ----------
bst = xgb.Booster()
bst.load_model(MODEL_PATH)

tfidf_word, tfidf_char = joblib.load(VECT_PATH)
numeric_cols = joblib.load(NUM_FEAT_PATH)
label_map = joblib.load(LABEL_MAP_PATH)
inv_label_map = {v: k for k, v in label_map.items()}

IDX_TO_LABEL = {0: "phishing", 1: "adult", 2: "legitimate"}

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
    """Feature extraction identical to evaluate_external.py"""
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

    adult_keywords = ("porn", "xxx", "sex", "adult", "cam", "tube", "nude",
                      "hot", "fuck", "escort", "babe", "boobs")
    suspicious_tlds = ("xyz", "top", "club", "info", "click", "link", "shop",
                       "work", "cf", "tk", "ml", "ga")

    num_digits = sum(1 for c in u2 if c.isdigit())
    digit_ratio = num_digits / max(1, len(u2))

    host_parts = host.split('.') if host else []
    suffix_parts = tx.suffix.split('.') if tx.suffix else []
    registered_parts = 1 + (len(suffix_parts) if suffix_parts else 0)
    num_subdomain_parts = max(0, len(host_parts) - registered_parts)

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
    }

    domain_path = (host + " " + path + " " + query).strip()
    return feats, domain_path, normalize_host(parsed.netloc)

# ---------- Prediction ----------
def predict_url(url):
    u = url.lower()
    adult_keywords = ("porn", "xxx", "sex", "adult", "cam", "tube",
                      "nude", "hot", "fuck", "escort", "babe", "boobs")

    feats, text, host_norm = extract_url_fields(url)

    # ✅ 1. Exact match whitelist
    whitelist_exact = {
        "youtube.com", "google.com", "facebook.com", "linkedin.com",
        "github.com", "twitter.com"
    }
    if host_norm in whitelist_exact:
        return {"phishing": 0.0, "adult": 0.0, "legitimate": 1.0}, 2, "legitimate", 1.0

    # ✅ 2. Direct adult keyword rule
    if any(k in u for k in adult_keywords):
        return {"phishing": 0.0, "adult": 1.0, "legitimate": 0.0}, 1, "adult", 1.0

    # ✅ 3. Model prediction
    num_vals = [feats.get(c, 0) for c in numeric_cols]
    X_num = sparse.csr_matrix([num_vals], dtype=np.float32)
    X_word = tfidf_word.transform([text])
    X_char = tfidf_char.transform([text])
    X_text = hstack([X_word, X_char])
    X = hstack([X_num, X_text], format="csr")

    d = xgb.DMatrix(X)
    probs = bst.predict(d)[0]

    final_idx = np.argmax(probs)
    pred_label = inv_label_map[final_idx]
    confidence = probs[final_idx]

    result_dict = {inv_label_map[i]: float(probs[i]) for i in range(len(probs))}
    return result_dict, final_idx, pred_label, confidence

# ---------- Main ----------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter URL: ").strip()

    probs, pred_idx, pred_label, confidence = predict_url(url)

    print("\nProbabilities:")
    for lbl, p in probs.items():
        print(f"  {lbl:<12} : {p:.3f}")

    print(f"\n✅ Predicted: {pred_label.upper()} (Confidence: {confidence:.3f})")
