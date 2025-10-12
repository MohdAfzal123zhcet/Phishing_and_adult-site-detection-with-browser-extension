# scripts/eval_compare.py
import pandas as pd, joblib, torch, numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

df = pd.read_csv("data/processed/url_features.csv")
df['is_phish'] = (df['label']=='phishing').astype(int)
feature_cols = [c for c in df.columns if c not in ("url","label","source","is_phish")]
X = df[feature_cols].fillna(0).values
y = df['is_phish'].values

# simple split same as training: use last 20% as test (if training used random_state 42)
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)

# XGBoost
xgb = joblib.load("model_service/models/pretrained/phishing_xgb.pkl")
# xgb may be sklearn wrapper or xgboost.Booster; try both
try:
    dtest = xgb.DMatrix(X_test)
    y_pred_xgb = xgb.predict(dtest)
except Exception:
    y_pred_xgb = xgb.predict_proba(X_test)[:,1]

# MLP
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,256),nn.ReLU(),nn.Dropout(0.2),
                                 nn.Linear(256,64),nn.ReLU(),nn.Dropout(0.2),
                                 nn.Linear(64,1))
    def forward(self,x): return self.net(x).squeeze(1)

model = MLP(X.shape[1])
model.load_state_dict(torch.load("model_service/models/own/phishing_mlp.pth", map_location='cpu'))
model.eval()
with torch.no_grad():
    y_pred_mlp = torch.sigmoid(model(torch.tensor(X_test).float())).numpy()

def report(y_true, y_scores, name):
    auc = roc_auc_score(y_true, y_scores)
    y_hat = (y_scores>=0.5).astype(int)
    p,r,f,_ = precision_recall_fscore_support(y_true, y_hat, average='binary', zero_division=0)
    print(f"{name}: AUC {auc:.4f} P {p:.3f} R {r:.3f} F1 {f:.3f}")

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
report(y_test, y_pred_xgb, "XGBoost")
report(y_test, y_pred_mlp, "MLP")
