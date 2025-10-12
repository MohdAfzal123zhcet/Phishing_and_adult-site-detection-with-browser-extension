# model_service/wrappers.py
import joblib, torch, numpy as np

class XGBWrapper:
    def __init__(self, path):
        self.model = joblib.load(path)
    def predict_score(self, features_list):
        try:
            import xgboost as xgb
            dm = xgb.DMatrix(features_list)
            return float(self.model.predict(dm)[0])
        except Exception:
            return float(self.model.predict_proba(np.array(features_list))[:,1][0])

class MLPWrapper:
    def __init__(self, path, in_dim):
        import torch.nn as nn
        class MLP(nn.Module):
            def __init__(self, in_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim,256), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(256,64), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(64,1)
                )
            def forward(self,x): return self.net(x).squeeze(1)
        self.model = MLP(in_dim)
        self.model.load_state_dict(torch.load(path, map_location='cpu'))
        self.model.eval()
    def predict_score(self, features_list):
        import torch
        X = torch.tensor(features_list, dtype=torch.float32)
        with torch.no_grad():
            out = torch.sigmoid(self.model(X)).numpy()
        return float(out[0])
