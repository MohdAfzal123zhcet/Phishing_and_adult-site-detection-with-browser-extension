# scripts/evaluate_external.py
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import sparse
from scipy.sparse import hstack   # ✅ added
from sklearn.metrics import classification_report, accuracy_score
from urllib.parse import urlparse
import tldextract
import re
from collections import Counter
import math

# ---------- Paths ----------
MODEL_PATH = "model_service/models/pretrained/multiclass_xgb.json"
VECT_PATH = "model_service/models/pretrained/tfidf_vectorizer.joblib"
NUM_FEAT_PATH = "model_service/models/pretrained/numeric_features.joblib"
LABEL_MAP_PATH = "model_service/models/pretrained/label_map.joblib"
DATA_PATH = "data/external/test_urls.csv"   # 👈 your external test CSV path

# ---------- Load Artifacts ----------
print("🔹 Loading model and vectorizer...")
bst = xgb.Booster()
bst.load_model(MODEL_PATH)

# ✅ Load both word + char TF-IDF vectorizers (tuple)
tfidf_word, tfidf_char = joblib.load(VECT_PATH)

numeric_cols = joblib.load(NUM_FEAT_PATH)

# --- Load and auto-fix label_map ---
label_map_raw = joblib.load(LABEL_MAP_PATH)
if all(isinstance(k, (int, float, np.integer, np.floating)) for k in label_map_raw.keys()):
    print("⚙️ Detected numeric keys in label_map — using training-time order (phishing=0, adult=1, legitimate=2)")
    label_map = {'phishing': 0, 'adult': 1, 'legitimate': 2}
elif all(isinstance(v, (str,)) for v in label_map_raw.values()):
    label_map = {v.lower().strip(): k for k, v in label_map_raw.items()}
else:
    label_map = {str(k).lower().strip(): v for k, v in label_map_raw.items()}

inv_label_map = {v: k for k, v in label_map.items()}

# ---------- Utility ----------
def entropy(s):
    if not s:
        return 0.0
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

    # same keyword sets as training
    adult_keywords = ("porn", "xxx", "sex", "adult", "cam", "tube",
                      "nude", "hot", "fuck", "escort", "babe", "boobs")
    phish_keywords = ("login", "signin", "verify", "update", "account",
                      "secure", "bank", "confirm", "wallet", "reset")

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
        "has_adult_keyword": int(any(k in u.lower() for k in adult_keywords)),
        "has_phish_keyword": int(any(k in u.lower() for k in phish_keywords)),
        "count_special_chars": sum(c in u for c in "@%=&?~"),
        "is_shortened_url": int(any(x in u.lower() for x in ("bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly"))),
        "token_count_path": path.count('/') + 1 if path else 0,
        "has_login_token": int(any(k in u.lower() for k in ("login", "signin", "verify", "update", "account", "secure"))),
        "tld_type": 0 if host.endswith((".gov", ".edu", ".org")) else 1
    }

    domain_path = (host + " " + path + " " + query).strip()
    return feats, domain_path


# ---------- Load External Data ----------
print("🔹 Loading external test data...")
df = pd.read_csv(DATA_PATH)
if 'url' not in df.columns or 'label' not in df.columns:
    raise ValueError("⚠️ CSV must contain 'url' and 'label' columns for evaluation.")

df['label'] = df['label'].astype(str).str.lower().str.strip()
valid_labels = set(label_map.keys())
invalid_rows = df[~df['label'].isin(valid_labels)]

if not invalid_rows.empty:
    print(f"⚠️ Skipping {len(invalid_rows)} rows with invalid labels: {invalid_rows['label'].unique()}")

df = df[df['label'].isin(valid_labels)].reset_index(drop=True)
if df.empty:
    raise ValueError("❌ No valid data found after filtering labels. Check your CSV label names!")

urls = df['url'].astype(str).tolist()
labels = df['label'].map(label_map).values

print(f"✅ Loaded {len(urls)} valid URLs for evaluation.")
print(f"Class distribution: {df['label'].value_counts().to_dict()}")

# ---------- Feature Extraction ----------
print("🔹 Extracting features...")
num_feat_list, text_list = [], []
for u in urls:
    feats, text = extract_url_fields(u)
    num_feat_list.append([feats.get(c, 0) for c in numeric_cols])
    text_list.append(text)

X_num = sparse.csr_matrix(num_feat_list, dtype=np.float32)

# ✅ Use both word + char TF-IDF vectors
X_word = tfidf_word.transform(text_list)
X_char = tfidf_char.transform(text_list)
X_text = hstack([X_word, X_char])

X = hstack([X_num, X_text], format="csr")

# ---------- Predict ----------
print("🔹 Predicting...")
dtest = xgb.DMatrix(X)
y_pred_proba = bst.predict(dtest)
y_pred = np.argmax(y_pred_proba, axis=1)

# ---------- Evaluate ----------
acc = accuracy_score(labels, y_pred)
print(f"\n✅ Custom Model Accuracy on External Test Set: {acc:.3f}\n")

print("📊 Classification Report:")
print(classification_report(labels, y_pred, target_names=list(label_map.keys())))

# ---------- Save Detailed Predictions ----------
results_df = pd.DataFrame({
    "url": urls,
    "true_label": [inv_label_map[i] for i in labels],
    "predicted_label": [inv_label_map[i] for i in y_pred],
    "confidence": [float(np.max(p)) for p in y_pred_proba]
})
results_df.to_csv("data/external/predictions_report.csv", index=False)
print("\n📝 Saved detailed results to: data/external/predictions_report.csv")
