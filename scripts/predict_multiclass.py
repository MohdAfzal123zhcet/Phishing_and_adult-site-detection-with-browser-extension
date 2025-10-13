# scripts/predict_multiclass.py
import joblib
import numpy as np
import re
from urllib.parse import urlparse
import tldextract
import xgboost as xgb
from scipy import sparse
from scipy.sparse import hstack   # ✅ Added for combining word + char vectors

MODEL_PATH = "model_service/models/pretrained/multiclass_xgb.json"
VECT_PATH = "model_service/models/pretrained/tfidf_vectorizer.joblib"
NUM_FEAT_PATH = "model_service/models/pretrained/numeric_features.joblib"
LABEL_MAP_PATH = "model_service/models/pretrained/label_map.joblib"

# ---------- Load Artifacts ----------
bst = xgb.Booster()
bst.load_model(MODEL_PATH)

# ✅ Load both word and char vectorizers (tuple saved during training)
tfidf_word, tfidf_char = joblib.load(VECT_PATH)

numeric_cols = joblib.load(NUM_FEAT_PATH)
label_map = joblib.load(LABEL_MAP_PATH)
inv_label_map = {v: k for k, v in label_map.items()}

# ---------- (NEW) index -> human label map ----------
IDX_TO_LABEL = {
    0: "phishing",
    1: "adult",
    2: "legitimate"
}

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

    # ✅ Always ensure scheme added (this fixes your mismatch)
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

    # ✅ same keyword sets as training
    adult_keywords = ("porn", "xxx", "sex", "adult", "cam", "tube", "nude", "hot", "fuck", "escort", "babe", "boobs")
    phish_keywords = ("login", "signin", "verify", "update", "account", "secure", "bank", "confirm", "wallet", "reset")

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
        "num_digits": sum(1 for c in u2 if c.isdigit()),
        "digit_ratio": sum(1 for c in u2 if c.isdigit()) / max(1, len(u2)),
        "entropy_domain": entropy(domain),
        "has_https": int(u2.lower().startswith("https")),
        "has_ip": int(bool(re.match(r'^\d+\.\d+\.\d+\.\d+$', tx.domain))),

        "has_adult_keyword": int(any(k in u2.lower() for k in adult_keywords)),
        "has_phish_keyword": int(any(k in u2.lower() for k in phish_keywords)),
        "count_special_chars": sum(c in u2 for c in "@%=&?~"),
        "is_shortened_url": int(any(x in u2.lower() for x in ("bit.ly","tinyurl","goo.gl","t.co","ow.ly"))),
        "token_count_path": path.count('/') + 1 if path else 0,
        "has_login_token": int(any(k in u2.lower() for k in ("login","signin","verify","update","account","secure"))),
        "tld_type": 0 if host.endswith((".gov",".edu",".org")) else 1
    }

    domain_path = (host + " " + path + " " + query).strip()
    return feats, domain_path



# ---------- Prediction ----------
def predict_url(url):
    feats, text = extract_url_fields(url)
    num_vals = [feats.get(c, 0) for c in numeric_cols]
    X_num = sparse.csr_matrix([num_vals], dtype=np.float32)

    # ✅ Use both word + char TF-IDF vectors (same as training)
    X_word = tfidf_word.transform([text])
    X_char = tfidf_char.transform([text])
    X_text = hstack([X_word, X_char])

    X = hstack([X_num, X_text], format="csr")

    d = xgb.DMatrix(X)
    probs = bst.predict(d)[0]  # length = num_classes

    out = {inv_label_map[i]: float(probs[i]) for i in range(len(probs))}
    pred_idx = np.argmax(probs)
    pred_label = inv_label_map[pred_idx]
    confidence = probs[pred_idx]

    return probs, out, pred_idx, pred_label, confidence


# ---------- Main ----------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter URL: ").strip()

    probs, out, pred_idx, pred_label, confidence = predict_url(url)

    print("\nProbabilities:")
    for i, p in enumerate(probs):
        human_label = IDX_TO_LABEL.get(i, inv_label_map.get(i, str(i)))
        print(f"  {i}: {p:.3f}  ->  {human_label}")

    pred_label_str = IDX_TO_LABEL.get(int(pred_idx), str(pred_label))
    print(f"\nPredicted: {pred_idx} -> {pred_label_str.upper()} (confidence={confidence:.3f})")
