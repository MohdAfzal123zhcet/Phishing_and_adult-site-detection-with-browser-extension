# scripts/fetch_adult_only_dataset.py
import requests
import pandas as pd
import csv
from pathlib import Path
from time import sleep

OUT = Path("data/processed/adult_urls_labeled.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

# List of trusted plain-text blocklist sources containing adult domains (domain-per-line)
SOURCES = [
    "https://raw.githubusercontent.com/blocklistproject/Lists/master/adult.txt",
    "https://raw.githubusercontent.com/chadmayfield/pihole-blocklists/master/lists/sex.txt",
    "https://raw.githubusercontent.com/blocklistproject/Lists/master/porn.txt",
    "https://raw.githubusercontent.com/blocklistproject/Lists/master/explicit.txt",
    # fallback community lists
    "https://raw.githubusercontent.com/StevenBlack/hosts/master/hosts",  # contains categories (will need filtering)
]

def fetch_txt(url):
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return r.text.splitlines()
    except Exception as e:
        print("Failed to fetch", url, ":", e)
        return []

domains = set()
for src in SOURCES:
    print("Fetching", src)
    lines = fetch_txt(src)
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        # common hosts files include "0.0.0.0 domain" or "127.0.0.1 domain"
        if ln.startswith("0.0.0.0 ") or ln.startswith("127.0.0.1 "):
            parts = ln.split()
            if len(parts) >= 2:
                dom = parts[1].lower()
            else:
                continue
        elif " " in ln and (ln.split()[0].count('.')==3):
            # weird host line
            dom = ln.split()[1].lower()
        else:
            dom = ln.lower()
        # basic cleanup
        dom = dom.lstrip("www.").strip()
        # skip IPs/non-domain tokens
        if any(c.isalpha() for c in dom):
            # remove URI fragments if present
            dom = dom.split("/")[0]
            domains.add(dom)
    sleep(0.5)

print("Unique candidate domains collected:", len(domains))

# If we still have <10000 domains, try expanding by combining sublists or repeating sources (rare)
if len(domains) < 10000:
    print("Only", len(domains), "domains found from sources. Trying extra community lists...")
    extra_sources = [
        "https://raw.githubusercontent.com/blocklistproject/Lists/master/adult-adservers.txt",
        "https://raw.githubusercontent.com/blocklistproject/Lists/master/erowid.txt"
    ]
    for src in extra_sources:
        lines = fetch_txt(src)
        for ln in lines:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            dom = ln.split()[0].lower()
            dom = dom.lstrip("www.").split("/")[0]
            if any(c.isalpha() for c in dom):
                domains.add(dom)
        sleep(0.5)
    print("After extras:", len(domains))

# Convert to list and trim/pad to 10000 if possible
domain_list = sorted(domains)
if len(domain_list) >= 10000:
    domain_list = domain_list[:10000]
else:
    # if still less than 10k, we'll keep what we have and warn user
    print("Warning: could not reach 10,000 unique domains with provided sources. Found:", len(domain_list))

# build dataframe with url,label
df = pd.DataFrame({
    "url": ["https://" + d for d in domain_list],
    "label": ["adult"] * len(domain_list)
})

# Save
df.to_csv(OUT, index=False, quoting=csv.QUOTE_MINIMAL)
print("Saved adult dataset:", OUT, "rows:", df.shape[0])
print(df.head(20).to_string(index=False))
