# scripts/merge_and_label.py
import os, pandas as pd, urllib.parse
os.makedirs("data/processed", exist_ok=True)
rows = []

# URLhaus
uh = "data/raw/urlhaus_urls.txt"
if os.path.exists(uh):
    with open(uh, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            u = line.strip()
            if not u or u.startswith("#"): continue
            rows.append({"url": u, "label": "phishing", "source": "urlhaus"})

# adult domains converted to urls
adultf = "data/raw/adult_domains_urls.txt"
if os.path.exists(adultf):
    with open(adultf, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            u = line.strip()
            if not u or u.startswith("#"): continue
            rows.append({"url": u, "label": "adult", "source": "adult_list"})

# seed CSV
seedcsv = "data/raw/seed_urls.csv"
if os.path.exists(seedcsv):
    try:
        df_seed = pd.read_csv(seedcsv)
        for _, r in df_seed.iterrows():
            rows.append({"url": str(r['url']), "label": str(r.get('label','benign')), "source": str(r.get('source','manual'))})
    except Exception as e:
        print("Could not read seed_urls.csv:", e)

# optional tranco list
tranco = "data/raw/tranco_sample.txt"
if os.path.exists(tranco):
    with open(tranco, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            dom = line.strip()
            if not dom: continue
            rows.append({"url": "http://"+dom, "label": "benign", "source": "tranco"})

df = pd.DataFrame(rows)
df = df.dropna(subset=['url']).drop_duplicates(subset=['url']).reset_index(drop=True)
def get_domain(u):
    try:
        h = urllib.parse.urlparse(u).hostname
        return h or ""
    except:
        return ""
df['domain'] = df['url'].apply(get_domain)
df['fetched_at'] = pd.Timestamp.utcnow().isoformat()
out = "data/processed/url_dataset.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows to {out}")
