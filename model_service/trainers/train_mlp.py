# model_service/trainers/train_mlp.py
import pandas as pd, os
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

os.makedirs("model_service/models/own", exist_ok=True)
df = pd.read_csv("data/processed/url_features.csv")
df['is_phish'] = (df['label']=='phishing').astype(int)
feature_cols = [c for c in df.columns if c not in ("url","label","source","is_phish")]
X = df[feature_cols].fillna(0).values.astype(np.float32)
y = df['is_phish'].values.astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)
tr = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=32, shuffle=True)
va = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=64)

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

best_loss = 1e9
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
    avg = sum(vals)/len(vals) if vals else 0
    print("Epoch", ep, "val_loss", avg)
    if avg < best_loss:
        best_loss = avg
        torch.save(model.state_dict(), "model_service/models/own/phishing_mlp.pth")
print("Saved best MLP with val_loss", best_loss)
