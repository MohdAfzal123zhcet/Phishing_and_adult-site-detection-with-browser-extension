# model_service/trainers/train_xgb.py
import pandas as pd, os, joblib
from sklearn.model_selection import train_test_split
import xgboost as xgb

os.makedirs("model_service/models/pretrained", exist_ok=True)
df = pd.read_csv("data/processed/url_features.csv")
df['is_phish'] = (df['label']=='phishing').astype(int)
feature_cols = [c for c in df.columns if c not in ("url","label","source","is_phish")]
X = df[feature_cols].fillna(0)
y = df['is_phish']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {"objective":"binary:logistic","eval_metric":"auc","max_depth":4,"eta":0.1}
bst = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtest,"test")], early_stopping_rounds=10)
joblib.dump(bst, "model_service/models/pretrained/phishing_xgb.pkl")
print("Saved XGBoost model to model_service/models/pretrained/phishing_xgb.pkl")
