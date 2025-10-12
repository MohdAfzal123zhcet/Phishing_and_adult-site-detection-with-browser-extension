import requests
import pandas as pd
import csv
from pathlib import Path
from io import StringIO

OUT = Path("data/processed/benign_top_legitimate.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

def fetch_tranco(n=10000):
    """
    Try to fetch Tranco top-1m list (current public mirrors).
    """
    urls = [
        "https://tranco-list.eu/top-1m.csv",  # old
        "https://tranco-list.eu/tranco_1m.csv",  # alternate
        "https://tranco-list.com/top-1m.csv",  # unofficial mirror
        "https://data.cyber-gordon.eu/tranco_1m.csv",  # mirror
    ]
    for u in urls:
        try:
            print("Trying Tranco mirror:", u)
            r = requests.get(u, timeout=20)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text), header=None, names=["rank", "domain"])
            domains = df["domain"].dropna().astype(str).tolist()
            if domains:
                print(f"✅ Tranco source OK → {len(domains)} domains")
                return domains[:n]
        except Exception as e:
            print("Tranco mirror failed:", e)
    return []

def fetch_majestic(n=10000):
    """
    Fetch from Majestic Million dataset (active in 2025).
    """
    try:
        url = "https://downloads.majestic.com/majestic_million.csv"
        print("Trying Majestic:", url)
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        if "Domain" in df.columns:
            domains = df["Domain"].dropna().astype(str).tolist()
            print(f"✅ Majestic source OK → {len(domains)} domains")
            return domains[:n]
    except Exception as e:
        print("Majestic failed:", e)
    return []

def fetch_github_backup(n=10000):
    """
    Backup list from verified top domain mirrors on GitHub.
    """
    urls = [
        "https://raw.githubusercontent.com/umpirsky/top-sites/master/top-sites.csv",
        "https://raw.githubusercontent.com/citizenlab/test-lists/master/lists/global.csv",
    ]
    for u in urls:
        try:
            print("Trying GitHub mirror:", u)
            r = requests.get(u, timeout=20)
            r.raise_for_status()
            text = r.text
            if "Domain" in text or "," in text:
                df = pd.read_csv(StringIO(text))
                if "Domain" in df.columns:
                    domains = df["Domain"].dropna().astype(str).tolist()
                elif "domain" in df.columns:
                    domains = df["domain"].dropna().astype(str).tolist()
                else:
                    domains = [line.split(",")[0].strip() for line in text.splitlines() if "." in line]
                print(f"✅ GitHub mirror OK → {len(domains)} domains")
                return domains[:n]
        except Exception as e:
            print("GitHub mirror failed:", e)
    return []

def main():
    N = 10000
    domains = fetch_tranco(N)
    if not domains:
        domains = fetch_majestic(N)
    if not domains:
        domains = fetch_github_backup(N)
    if not domains:
        raise SystemExit("❌ All sources failed — check internet or try again later.")

    # clean duplicates
    domains = [d.strip().lstrip("www.") for d in domains if isinstance(d, str) and d.strip()]
    domains = list(dict.fromkeys(domains))  # remove duplicates, keep order

    df = pd.DataFrame({
        "url": ["https://" + d for d in domains],
        "label": ["legitimate"] * len(domains)
    })
    df.to_csv(OUT, index=False, quoting=csv.QUOTE_MINIMAL)
    print("✅ Saved benign file:", OUT, "rows:", df.shape[0])

if __name__ == "__main__":
    main()
