# scripts/fetch_phishing_dataset.py
import requests
import pandas as pd
import csv
import os
import time
from pathlib import Path
from urllib.parse import urlparse

OUT = Path("data/processed/phishing_legit_combined.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

# --- public plain-text / raw-list sources that usually contain phishing domains or malicious URLs ---
SOURCES_TXT = [
    "https://raw.githubusercontent.com/blocklistproject/Lists/master/phishing.txt",
    "https://raw.githubusercontent.com/blocklistproject/Lists/master/phishing-domains.txt",  # sometimes present
    # fallback community lists (may contain mixed malware/phishing)
    "https://raw.githubusercontent.com/stamparm/blackbook/master/malicious_domains.txt",
]

# --- CSV sources (URLhaus) that contain malicious URLs (URLhaus returns csv of URLs) ---
SOURCES_CSV = [
    "https://urlhaus.abuse.ch/downloads/csv/",  # URLhaus: list of malicious URLs (includes many phishing)
]

# Optional: PhishTank requires an API key to download the official dataset.
# If you have one, set environment variable PHISHTANK_KEY to use it (optional).
PHISHTANK_KEY = os.environ.get("PHISHTANK_KEY")  # optional

def fetch_text_lines(url, timeout=20):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.text.splitlines()
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return []

def extract_domain_from_line(line):
    # line may be "0.0.0.0 domain" or just domain or a full URL
    ln = line.strip()
    if not ln or ln.startswith("#"):
        return None
    parts = ln.split()
    # if it's a hosts-style line like "0.0.0.0 domain"
    if len(parts) >= 2 and (parts[0] == "0.0.0.0" or parts[0] == "127.0.0.1"):
        dom = parts[1]
    else:
        dom = parts[0]
    # remove scheme if present
    if dom.startswith("http://") or dom.startswith("https://"):
        try:
            dom = urlparse(dom).netloc
        except:
            pass
    dom = dom.strip().lstrip("www.").split("/")[0]
    # basic validation
    if "." in dom and any(c.isalpha() for c in dom):
        return dom.lower()
    return None

# collect domains
phish_domains = set()

print("Fetching text-based lists...")
for src in SOURCES_TXT:
    lines = fetch_text_lines(src)
    for ln in lines:
        dom = extract_domain_from_line(ln)
        if dom:
            phish_domains.add(dom)
    time.sleep(0.5)

# fetch URLhaus CSV (if reachable)
print("Fetching URLhaus CSV (may contain many malicious URLs including phishing)...")
for src in SOURCES_CSV:
    try:
        r = requests.get(src, timeout=30)
        r.raise_for_status()
        text = r.text
        # parse CSV lines, find URL column (URLhaus format: url,status_date,threat, tags, ...)
        for row in csv.reader(text.splitlines()):
            if not row:
                continue
            # try to find a column that looks like a url
            for cell in row:
                if cell.startswith("http://") or cell.startswith("https://") or "www." in cell:
                    dom = urlparse(cell).netloc.lstrip("www.").split(":")[0]
                    if dom:
                        phish_domains.add(dom.lower())
                        break
        time.sleep(0.5)
    except Exception as e:
        print("URLhaus fetch failed:", e)

# optional: PhishTank (requires key)
if PHISHTANK_KEY:
    try:
        print("Fetching PhishTank (using PHISHTANK_KEY from env)...")
        phishtank_url = f"https://data.phishtank.com/data/online-valid.csv?key={PHISHTANK_KEY}"
        r = requests.get(phishtank_url, timeout=30)
        r.raise_for_status()
        for row in csv.reader(r.text.splitlines()):
            # common PhishTank CSV has URL in first or second column; search cells for http
            for cell in row:
                if cell.startswith("http://") or cell.startswith("https://"):
                    dom = urlparse(cell).netloc.lstrip("www.").split(":")[0]
                    if dom:
                        phish_domains.add(dom.lower())
                        break
        print("PhishTank fetched OK.")
    except Exception as e:
        print("PhishTank fetch failed or API key not valid:", e)

print(f"Collected phishing candidate domains: {len(phish_domains):,}")

# If not enough domains, warn
if len(phish_domains) < 5000:
    print("Warning: fewer than 5k candidate domains found. You may want to add more sources or provide a local phishing list.")

# Build phishing DataFrame (limit to 10000)
phish_list = sorted(phish_domains)
if len(phish_list) >= 10000:
    phish_list = phish_list[:10000]

df_phish = pd.DataFrame({
    "url": ["https://" + d for d in phish_list],
    "label": ["phishing"] * len(phish_list)
})
print("Phishing rows:", df_phish.shape[0])

# --- Get benign (top sites) from Tranco / common list
print("Fetching benign top domains (Tranco top) for balance...")
try:
    tranco_url = "https://tranco-list.eu/top-1m.csv"
    top = pd.read_csv(tranco_url, header=None, names=["rank", "domain"]).head(len(df_phish))
    df_benign = pd.DataFrame({
        "url": ["https://" + d for d in top["domain"].astype(str)],
        "label": ["benign"] * len(top)
    })
    print("Benign rows:", df_benign.shape[0])
except Exception as e:
    print("Tranco fetch failed:", e)
    # fallback: build a small benign list
    benign_sample = ["google.com", "youtube.com", "facebook.com", "amazon.com", "wikipedia.org", "twitter.com", "instagram.com"]
    df_benign = pd.DataFrame({
        "url": ["https://" + d for d in benign_sample],
        "label": ["benign"] * len(benign_sample)
    })

# combine, shuffle
combined = pd.concat([df_phish, df_benign], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# save
combined.to_csv(OUT, index=False, quoting=csv.QUOTE_MINIMAL)
print("Saved combined dataset:", OUT, "rows:", combined.shape[0])
print(combined['label'].value_counts())
