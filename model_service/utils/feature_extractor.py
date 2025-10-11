# model_service/utils/feature_extractor.py
import re
from urllib.parse import urlparse
from collections import Counter
import math

SUSP = re.compile(r"(login|verify|secure|account|update|bank|confirm|signin|adult|porn|xxx|sex)", re.I)

def entropy(s: str) -> float:
    if not s:
        return 0.0
    c = Counter(s)
    probs = [v/len(s) for v in c.values()]
    return -sum(p * math.log2(p) for p in probs)

def has_ip(host: str) -> int:
    return 1 if re.match(r"^\d+\.\d+\.\d+\.\d+$", host or "") else 0

def extract(url: str) -> dict:
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    q = parsed.query or ""
    features = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "num_subdomains": hostname.count("."),
        "num_digits": sum(c.isdigit() for c in url),
        "has_ip": has_ip(hostname),
        "has_at": 1 if "@" in url else 0,
        "num_params": 0 if not q else len(q.split("&")),
        "suspicious_tokens": 1 if SUSP.search(url) else 0,
        "entropy": round(entropy(hostname + path), 4),
        "has_https": 1 if parsed.scheme == "https" else 0,
        "punycode": 1 if "xn--" in hostname else 0
    }
    return features
