# app.py
import os
import io
import json
import requests
from urllib.parse import urljoin
from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
from nudenet import NudeDetector

# ---------- CONFIG ----------
MAX_IMAGES = 3
ADULT_KEYWORDS = [
    "porn","sex","xxx","boobs","fuck","nude","cam","escort",
    "adult","nsfw","hot","babe","18+","naked","hardcore"
]
detector = NudeDetector()          # loads model on first call
app = Flask(__name__)
CORS(app)                           # allow browser -> server requests

# ---------- helpers ----------
def extract_text_and_images_from_html(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True).lower()
    imgs = []
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if src:
            imgs.append(urljoin(base_url, src))
    for vid in soup.find_all("video"):
        if vid.get("poster"):
            imgs.append(urljoin(base_url, vid["poster"]))
    return text, list(dict.fromkeys(imgs))

def text_adult_score(text):
    hits = sum(text.count(k) for k in ADULT_KEYWORDS)
    return min(hits / 5.0, 1.0)

def image_adult_score_from_urls(img_urls):
    scores = []
    headers = {"User-Agent":"Mozilla/5.0"}
    for u in img_urls[:MAX_IMAGES]:
        try:
            r = requests.get(u, timeout=8, headers=headers)
            ctype = r.headers.get("Content-Type","")
            if not r.ok or "image" not in ctype:
                continue
            tmp_name = "tmp_img_" + os.path.basename(u.split("?")[0] or "img.jpg")
            with open(tmp_name, "wb") as f:
                f.write(r.content)
            detections = detector.detect(tmp_name)
            # detector.detect returns list of dicts (boxes/classes). score mapping heuristic:
            if any(d.get("class") in (
                "EXPOSED_BREAST_F","EXPOSED_GENITALIA_F","EXPOSED_GENITALIA_M",
                "EXPOSED_BUTTOCKS","EXPOSED_ANUS") for d in detections):
                scores.append(1.0)
            elif any(d.get("class") == "EXPOSED_BELLY" for d in detections):
                scores.append(0.5)
            else:
                scores.append(0.0)
            os.remove(tmp_name)
        except Exception:
            continue
    return max(scores) if scores else 0.0

# ---------- API ----------
@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accepts JSON:
    {
      "html": "<full html string>",
      "base_url": "https://example.com"
    }
    Returns JSON with text_score, image_score, adult_score, decision.
    """
    data = request.get_json(force=True)
    html = data.get("html", "")
    base_url = data.get("base_url", data.get("url", ""))
    if not html or not base_url:
        return jsonify({"error": "missing html or base_url"}), 400

    text, imgs = extract_text_and_images_from_html(html, base_url)
    tscore = text_adult_score(text)
    iscore = image_adult_score_from_urls(imgs)
    adult_score = 0.7 * iscore + 0.3 * tscore
    label = (
        "adult" if adult_score >= 0.75
        else "suspect" if adult_score >= 0.4
        else "safe"
    )
    result = {
        "text_score": round(tscore,3),
        "image_score": round(iscore,3),
        "adult_score": round(adult_score,3),
        "decision": label,
        "num_images": len(imgs),
        "images": imgs[:MAX_IMAGES]
    }
    return jsonify(result)

if __name__ == "__main__":
    # server listens on 0.0.0.0 port 5000; change if needed
    app.run(host="0.0.0.0", port=5000, debug=False)
