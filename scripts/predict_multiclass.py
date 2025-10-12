# scripts/predict_multiclass.py
import joblib
import numpy as np
import re
from urllib.parse import urlparse
import tldextract
import xgboost as xgb
from scipy import sparse

MODEL_PATH = "model_service/models/pretrained/multiclass_xgb.json"
VECT_PATH = "model_service/models/pretrained/tfidf_vectorizer.joblib"
NUM_FEAT_PATH = "model_service/models/pretrained/numeric_features.joblib"
LABEL_MAP_PATH = "model_service/models/pretrained/label_map.joblib"

# ---------- Load Artifacts ----------
bst = xgb.Booster()
bst.load_model(MODEL_PATH)
tfidf = joblib.load(VECT_PATH)
numeric_cols = joblib.load(NUM_FEAT_PATH)
label_map = joblib.load(LABEL_MAP_PATH)
inv_label_map = {v: k for k, v in label_map.items()}

# ---------- Utility ----------
def entropy(s):
    if not s:
        return 0.0
    from collections import Counter
    import math
    probs = [v / len(s) for v in Counter(s).values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)

def extract_url_fields(url: str):
    u = str(url).strip()
    if not u.startswith(("http://", "https://")):
        u2 = "http://" + u
    else:
        u2 = u
    parsed = urlparse(u2)
    tx = tldextract.extract(u2)
    domain = tx.domain or ""
    subdomain = tx.subdomain or ""
    path = parsed.path or ""
    query = parsed.query or ""
    host = parsed.netloc.lower()

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
    domain_path = (host + " " + path + " " + query).strip()
    return feats, domain_path

# ---------- Prediction ----------
def predict_url(url):
    feats, text = extract_url_fields(url)
    num_vals = [feats.get(c, 0) for c in numeric_cols]
    X_num = sparse.csr_matrix([num_vals], dtype=np.float32)
    X_text = tfidf.transform([text])
    X = sparse.hstack([X_num, X_text], format="csr")

    d = xgb.DMatrix(X)
    probs = bst.predict(d)[0]  # length = num_classes

    # Map probabilities back to class names
    out = {inv_label_map[i]: float(probs[i]) for i in range(len(probs))}
    pred_idx = np.argmax(probs)
    pred_label = inv_label_map[pred_idx]
    confidence = probs[pred_idx]

    return out, pred_label, confidence

# ---------- Main ----------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter URL: ").strip()

    probs, pred_label, confidence = predict_url(url)

    print("\nProbabilities:")
    for k, v in probs.items():
        print(f"  {k}: {v:.3f}")

print(f"\nPredicted: {str(pred_label).upper()} (confidence={confidence:.3f})")
