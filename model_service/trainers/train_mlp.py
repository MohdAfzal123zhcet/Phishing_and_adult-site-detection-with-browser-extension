# model_service/trainers/train_mlp.py
import os
import pandas as pd
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

os.makedirs("model_service/models/own", exist_ok=True)
DATA_PATH = "data/processed/url_features.csv"   # change if you use url_features_clean.csv
MODEL_OUT = "model_service/models/own/phishing_mlp.pth"

print("Loading:", DATA_PATH)
df = pd.read_csv(DATA_PATH, on_bad_lines='skip', quoting=3)
df.columns = df.columns.str.strip()

# accept 'type' -> 'label' if needed
if 'label' not in df.columns and 'type' in df.columns:
    df = df.rename(columns={'type':'label'})

if 'label' not in df.columns:
    raise SystemExit("Dataset must have a 'label' or 'type' column.")

df['label'] = df['label'].astype(str).str.strip().str.lower()
df['is_phish'] = df['label'].apply(lambda x: 1 if x in ["phishing", "malicious", "phish"] else 0)

# Select numeric feature columns already present
feature_cols = [c for c in df.columns if c not in ("url","label","source","is_phish")]
numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

# If no numeric features found, create basic URL features (same as XGB)
if len(numeric_features) == 0:
    print("No numeric features found — creating basic URL features...")
    from urllib.parse import urlparse
    import re

    def extract_basic(url: str):
        u = str(url).strip()
        parsed = urlparse(u if '://' in u else 'http://' + u)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        query = parsed.query.lower()
        return {
            "url_length": len(u),
            "domain_length": len(domain),
            "path_length": len(path),
            "num_dots": domain.count('.') + path.count('.'),
            "num_digits": len(re.findall(r'\d', u)),
            "has_https": int(u.lower().startswith("https")),
            "has_porn": int(any(k in u.lower() for k in ["porn","sex","xxx","adult"]))
        }

    basic = df['url'].astype(str).apply(extract_basic).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), basic.reset_index(drop=True)], axis=1)
    numeric_features = basic.columns.tolist()

print("Numeric features to use:", numeric_features)
print("Sample rows:", df.shape[0])

# Convert to arrays
X = df[numeric_features].fillna(0).values.astype(np.float32)
y = df['is_phish'].astype(np.float32).values

print("Feature matrix shape:", X.shape)

if X.shape[1] == 0:
    raise SystemExit("No input features available for training (X has 0 columns). Fix feature extraction.")

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tr = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=32, shuffle=True)
va = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=64)

# MLP
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
model = MLP(X.shape[1]).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

best_loss = float("inf")
for ep in range(30):
    model.train()
    for xb,yb in tr:
        xb,yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()

    # val
    model.eval()
    vals = []
    with torch.no_grad():
        for xb,yb in va:
            xb,yb = xb.to(device), yb.to(device)
            vals.append(float(loss_fn(model(xb), yb)))
    avg = sum(vals)/len(vals) if vals else 0.0
    print(f"Epoch {ep} val_loss {avg:.6f}")

    # always overwrite best; also save at end to guarantee proper file
    if avg < best_loss:
        best_loss = avg
        torch.save(model.state_dict(), MODEL_OUT)
        print("Saved best MLP with val_loss", best_loss)

# final save to guarantee file updated
torch.save(model.state_dict(), MODEL_OUT)
print("Final model saved at", MODEL_OUT, "with val_loss", best_loss)
