# model_service/registry.py
import os, pandas as pd
from model_service.wrappers import XGBWrapper, MLPWrapper
from model_service.utils.feature_extractor import extract

class ModelRegistry:
    def __init__(self):
        self._models = {}
        self.active = None

    def load_all(self):
        xgb_path = "model_service/models/pretrained/phishing_xgb.pkl"
        if os.path.exists(xgb_path):
            self._models['pretrained_xgb'] = XGBWrapper(xgb_path)
        mlp_path = "model_service/models/own/phishing_mlp.pth"
        if os.path.exists(mlp_path):
            df = pd.read_csv("data/processed/url_features.csv")
            feature_cols = [c for c in df.columns if c not in ("url","label","source")]
            in_dim = len(feature_cols)
            self._models['own_mlp'] = MLPWrapper(mlp_path, in_dim)
        if not self._models:
            # fallback: create a trivial scoreer that warns nothing (avoid crash)
            self._models['noop'] = lambda *args, **kwargs: 0.0
            self.active = 'noop'
        else:
            self.active = list(self._models.keys())[0]

    def names(self):
        return list(self._models.keys())

    def set_active(self, name):
        if name not in self._models:
            raise KeyError("Model not found")
        self.active = name

    def score_url(self, url):
        feats = extract(url)
        # preserve original feature order as in processed CSV
        try:
            df = pd.read_csv("data/processed/url_features.csv", nrows=1)
            feature_cols = [c for c in df.columns if c not in ("url","label","source")]
            vector = [feats.get(c, 0) for c in feature_cols]
        except Exception:
            # fallback ordering
            vector = list(feats.values())
        # if active is a callable (noop)
        model = self._models[self.active]
        if callable(model):
            return 0.0, self.active
        score = model.predict_score([vector])
        return score, self.active
