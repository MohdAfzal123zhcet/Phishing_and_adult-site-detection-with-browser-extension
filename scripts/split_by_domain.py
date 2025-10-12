import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

print("🔹 Loading feature dataset...")
df = pd.read_csv("data/processed/url_features.csv")

# confirm label column name
label_col = "label"
if label_col not in df.columns:
    raise ValueError(f"❌ 'label' column not found in dataset. Columns: {df.columns.tolist()}")

X = df.drop(columns=[label_col])
y = df[label_col]

# Remove rare classes (<2 samples)
counts = Counter(y)
rare_classes = [cls for cls, c in counts.items() if c < 2]
if rare_classes:
    print("⚠️ Removing rare classes:", rare_classes)
    mask = ~y.isin(rare_classes)
    X = X[mask]
    y = y[mask]

print("✅ Classes after cleaning:", Counter(y))

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save splits
X_test.to_csv("data/processed/X_test.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print(f"✅ Test split created successfully! X_test shape: {X_test.shape}")
