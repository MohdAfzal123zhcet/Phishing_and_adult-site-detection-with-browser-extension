# scripts/test_mlp_model.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import re

# ---------- same features as training ----------
def extract_basic(url: str):
    u = str(url).strip()
    parsed = urlparse(u if '://' in u else 'http://' + u)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    return {
        "url_length": len(u),
        "domain_length": len(domain),
        "path_length": len(path),
        "num_dots": domain.count('.') + path.count('.'),
        "num_digits": len(re.findall(r'\d', u)),
        "has_https": int(u.lower().startswith("https")),
        "has_porn": int(any(k in u.lower() for k in ["porn","sex","xxx","adult"]))
    }

# ---------- model ----------
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256,64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,1)
        )
    def forward(self,x): return self.net(x).squeeze(1)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- load model ----------
model_path = "model_service/models/own/phishing_mlp.pth"
model = MLP(7).to(device)  # <-- 7 features only
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ Loaded MLP model")

# ---------- test loop ----------
while True:
    url = input("🔗 Enter URL to test (or 'exit'): ").strip()
    if url.lower() == "exit":
        break

    feats = extract_basic(url)
    X = np.array([[feats[f] for f in ["url_length","domain_length","path_length","num_dots","num_digits","has_https","has_porn"]]], dtype=np.float32)
    X_tensor = torch.tensor(X).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(X_tensor)).item()

    label = "PHISHING" if prob >= 0.5 else "LEGITIMATE"
    print(f"Prediction: {label} (confidence={prob:.3f})\n")
