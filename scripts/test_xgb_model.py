import joblib
import pandas as pd
from urllib.parse import urlparse
import xgboost as xgb

# Load trained Booster
model = joblib.load("model_service/models/pretrained/phishing_xgb.pkl")
print("✅ Loaded XGBoost model")

# --- feature extractor (same names & same order as training) ---
def extract_url_features(url):
    parsed = urlparse(url)
    host = parsed.netloc or ""
    path = parsed.path or ""

    features = {
        "is_phish": 0,  # dummy (only needed for order)
        "url_length": len(url),
        "domain_length": len(host),
        "path_length": len(path),
        "num_dots": url.count('.'),
        "num_digits": sum(c.isdigit() for c in url),
        "has_https": int(url.startswith("https")),
        "has_porn": int(any(word in url.lower() for word in ["porn", "xxx", "adult", "sex"]))
    }

    # ensure exact column order as during training
    ordered = [
        "is_phish", "url_length", "domain_length", "path_length",
        "num_dots", "num_digits", "has_https", "has_porn"
    ]
    return pd.DataFrame([[features[c] for c in ordered]], columns=ordered)

# --- Input URL ---
url = input("🔗 Enter URL to test: ").strip()
feat = extract_url_features(url)
dtest = xgb.DMatrix(feat)

# --- Predict ---
pred_prob = float(model.predict(dtest)[0])
pred_label = "phishing" if pred_prob > 0.5 else "legitimate"

print(f"\nPrediction: {pred_label.upper()} (confidence={pred_prob:.3f})")
