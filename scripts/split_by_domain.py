# scripts/split_by_domain.py
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import os

df = pd.read_csv("data/processed/url_features.csv")
if 'domain' not in df.columns:
    df['domain'] = df['url'].apply(lambda u: __import__('urllib.parse').urlparse(u).hostname or "")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['domain']))
train = df.iloc[train_idx].reset_index(drop=True)
test = df.iloc[test_idx].reset_index(drop=True)
os.makedirs("data/processed/splits", exist_ok=True)
train.to_csv("data/processed/splits/train.csv", index=False)
test.to_csv("data/processed/splits/test.csv", index=False)
print("Train rows:", len(train), "Test rows:", len(test))
