#!/usr/bin/env python3
"""
Populate labeled URL CSVs from public feeds.

Outputs:
 - data/datasets/phishing_urls_labeled.csv
 - data/datasets/adult_urls_labeled.csv
 - data/datasets/legitimate_urls_labeled.csv
 - data/datasets/dataset_urls_labels.csv (combined, deduped; precedence phishing>adult>legit)
 - Also writes seed txt files in data/datasets/*.txt

Note:
 - Homepage URL format used: https://<domain> (no path)
 - If you want internal page URLs, run crawler separately.
"""

import os, re, time, csv, requests
from tranco import Tranco

OUT_DIR = "data/datasets"
os.makedirs(OUT_DIR, exist_ok=True)

# sources (you can expand these lists)
PHISHING_SOURCES = [
    "https://openphish.com/feed.txt",
    "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-db.txt",
    "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt",
    "https://raw.githubusercontent.com/MetaMask/eth-phishing-detect/master/src/hosts.json"
]

ADULT_SOURCES = [
    "https://raw.githubusercontent.com/boncey/Adult-Porn-Urls/master/adult-domains.txt",
    "https://raw.githubusercontent.com/blocklistproject/Lists/master/adult.txt",
    "https://raw.githubusercontent.com/chadmayfield/porn-domains/master/block.txt",
    "https://raw.githubusercontent.com/azet/inspektor/master/blacklists/adult",
    "https://raw.githubusercontent.com/StevenBlack/hosts/master/hosts"
]

# configurable: set to int to cap per-class results, or None to keep all
TARGET_LIMIT_PER_CLASS = None  # e.g., 30000

HEADERS = {"User-Agent": "SecureBrowseSeedCollector/1.0"}

_domain_re = re.compile(r"(?:https?://)?([^/:\s]+)", flags=re.I)
_extract_domains_re = re.compile(r"(?:https?://)?([a-z0-9\-\._]+\.[a-z]{2,})", flags=re.I)

def normalize_domain(candidate: str) -> str:
    if not candidate:
        return ""
    token = candidate.strip().lower()
    m = _domain_re.match(token)
    host = m.group(1) if m else token
    host = host.split(':')[0].strip().rstrip('/')
    # basic validation
    if host and any(c.isalnum() for c in host) and len(host) <= 253:
        return host
    return ""

def domain_to_homepage(domain: str) -> str:
    return f"https://{domain}"

def fetch_text(url, timeout=20):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r.text
        else:
            print(f"Warning: {url} returned HTTP {r.status_code}")
    except Exception as e:
        print(f"Warning: failed to fetch {url} -> {e}")
    return ""

def collect_domains_from_sources(sources):
    collected = []
    seen = set()
    for src in sources:
        print("Fetching", src)
        txt = fetch_text(src)
        if not txt:
            continue
        # common formats: hosts files (0.0.0.0 domain), plain domain lists, or raw urls
        for line in txt.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            token = parts[-1]  # usually the domain
            d = normalize_domain(token)
            if d and d not in seen:
                seen.add(d); collected.append(d)
        # also try to find any urls/domains embedded in free text
        for m in _extract_domains_re.findall(txt):
            d = normalize_domain(m)
            if d and d not in seen:
                seen.add(d); collected.append(d)
        time.sleep(0.2)
    return collected

def fetch_tranco_top(n=100000):
    print("Fetching Tranco top list...")
    try:
        tr = Tranco().list()
        entries = list(tr.top(n))
        out = []
        seen = set()
        for e in entries:
            d = normalize_domain(e)
            if d and d not in seen:
                seen.add(d); out.append(d)
        return out
    except Exception as e:
        print("Tranco fetch failed:", e)
        return []

def write_seed_txt(path, domain_list):
    with open(path, "w", encoding="utf-8") as f:
        for d in domain_list:
            f.write(d + "\n")
    print("Wrote", path, len(domain_list))

def write_labeled_csv(path, domain_list, label, limit=None):
    if limit:
        domain_list = domain_list[:limit]
    rows = [(domain_to_homepage(d), label) for d in domain_list]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url","label"])
        for u,l in rows:
            writer.writerow([u,l])
    print("Wrote", path, len(rows))
    return rows

def main():
    # phishing
    phishing_domains = collect_domains_from_sources(PHISHING_SOURCES)
    write_seed_txt(os.path.join(OUT_DIR, "phishing_seed_list.txt"), phishing_domains)
    ph_rows = write_labeled_csv(os.path.join(OUT_DIR, "phishing_urls_labeled.csv"), phishing_domains, "phishing", limit=TARGET_LIMIT_PER_CLASS)

    # adult
    adult_domains = collect_domains_from_sources(ADULT_SOURCES)
    write_seed_txt(os.path.join(OUT_DIR, "adult_seed_list.txt"), adult_domains)
    ad_rows = write_labeled_csv(os.path.join(OUT_DIR, "adult_urls_labeled.csv"), adult_domains, "adult", limit=TARGET_LIMIT_PER_CLASS)

    # legit from Tranco
    legit_domains = fetch_tranco_top(n=200000)
    write_seed_txt(os.path.join(OUT_DIR, "legit_seed_list.txt"), legit_domains)
    lg_rows = write_labeled_csv(os.path.join(OUT_DIR, "legitimate_urls_labeled.csv"), legit_domains, "legitimate", limit=TARGET_LIMIT_PER_CLASS)

    # combined with precedence (phishing > adult > legitimate)
    combined_map = {}
    for u,l in ph_rows:
        combined_map[u] = l
    for u,l in ad_rows:
        if u not in combined_map:
            combined_map[u] = l
    for u,l in lg_rows:
        if u not in combined_map:
            combined_map[u] = l

    combined_list = list(combined_map.items())
    print("Total combined unique urls:", len(combined_list))
    # optionally limit total combined or shuffle
    # write combined CSV
    with open(os.path.join(OUT_DIR, "dataset_urls_labels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url","label"])
        for u,l in combined_list:
            w.writerow([u,l])
    print("Wrote combined dataset_urls_labels.csv")

if __name__ == "__main__":
    main()
