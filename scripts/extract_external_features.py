import pandas as pd
import numpy as np
import tldextract, re, math, os
from urllib.parse import urlparse
import joblib
from scipy import sparse

# =========================
# 1️⃣ Load external test data
# =========================
df = pd.read_csv("data/external/test_urls.csv")
print("🔹 Loaded external data:", df.shape)

# =========================
# 2️⃣ Load vectorizer + numeric feature list
# =========================
VECT_PATH = "model_service/models/pretrained/tfidf_vectorizer.joblib"
NUM_FEAT_PATH = "model_service/models/pretrained/numeric_features.joblib"

tfidf = joblib.load(VECT_PATH)
numeric_cols = joblib.load(NUM_FEAT_PATH)

# =========================
# 3️⃣ Define feature extractor
# =========================
def entropy(s):
    if not s:
        return 0.0
    from collections import Counter
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

# =========================
# 4️⃣ Extract numeric + text features
# =========================
num_list, text_list = [], []

for url in df["url"]:
    feats, text = extract_url_fields(url)
    num_vals = [feats.get(c, 0) for c in numeric_cols]
    num_list.append(num_vals)
    text_list.append(text)

X_num = sparse.csr_matrix(num_list, dtype=np.float32)
X_text = tfidf.transform(text_list)
X_combined = sparse.hstack([X_num, X_text], format="csr")

print("✅ Combined feature matrix shape:", X_combined.shape)

# =========================
# 5️⃣ Save split versions
# =========================
os.makedirs("data/processed", exist_ok=True)
joblib.dump(X_combined, "data/processed/X_test_external.joblib")
df["label"].to_csv("data/processed/y_test_external.csv", index=False)
print("✅ Saved X_test_external.joblib and y_test_external.csv successfully!")
