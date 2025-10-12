import pandas as pd
import csv
from pathlib import Path

# file paths
files = [
    "data/processed/benign_top_legitimate.csv",
    "data/processed/phishing_legit_combined.csv",
    "data/processed/adult_urls_labeled.csv"
]

output_path = Path("data/processed/url_features_clean.csv")
merged = []

for file in files:
    try:
        df = pd.read_csv(file, on_bad_lines='skip', quoting=csv.QUOTE_MINIMAL)
        df.columns = df.columns.str.strip().str.lower()
        # handle cases like "type" instead of "label"
        if 'type' in df.columns and 'label' not in df.columns:
            df.rename(columns={'type': 'label'}, inplace=True)
        # if single-column file like "url,label" text merged
        if df.shape[1] == 1 and 'url' not in df.columns:
            temp = df.iloc[:, 0].astype(str).str.split(',', n=1, expand=True)
            temp.columns = ['url', 'label']
            df = temp
        # keep only 2 columns
        if 'url' in df.columns and 'label' in df.columns:
            df = df[['url', 'label']]
            merged.append(df)
            print(f"✅ Loaded: {file} → {len(df)} rows")
        else:
            print(f"⚠️ Skipped (missing columns): {file}")
    except Exception as e:
        print(f"❌ Error reading {file}: {e}")

# combine all
if not merged:
    raise SystemExit("No valid data found.")

final_df = pd.concat(merged, ignore_index=True)

# remove duplicates
final_df.drop_duplicates(subset='url', inplace=True)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# save
output_path.parent.mkdir(parents=True, exist_ok=True)
final_df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
print(f"\n💾 Merged dataset saved → {output_path}")
print("Rows:", len(final_df))
print(final_df['label'].value_counts())
