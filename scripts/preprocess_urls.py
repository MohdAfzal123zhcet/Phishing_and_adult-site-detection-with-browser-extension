# scripts/preprocess_chunked.py
import pandas as pd
from model_service.utils.feature_extractor import extract
import os

IN = "data/processed/url_dataset.csv"   # merged raw urls csv
OUT = "data/processed/url_features_chunked.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

chunksize = 5000   # adjust down (1000) if RAM is low
first = True

for i, df_chunk in enumerate(pd.read_csv(IN, chunksize=chunksize)):
    print(f"Processing chunk {i} rows {len(df_chunk)}")
    rows = []
    for url in df_chunk['url'].astype(str):
        feats = extract(url)
        feats['url'] = url
        rows.append(feats)
    out_df = pd.DataFrame(rows)
    if first:
        out_df.to_csv(OUT, index=False, mode='w', encoding='utf-8')
        first = False
    else:
        out_df.to_csv(OUT, index=False, mode='a', header=False, encoding='utf-8')
    print(f"Appended chunk {i} -> {OUT}")
print("Done.")
