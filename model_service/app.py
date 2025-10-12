# model_service/app.py
import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from model_service.registry import ModelRegistry
from model_service.persistence import save_active, load_active
from model_service.auth import require_admin_key

app = FastAPI(title="SecureBrowse URL-only service")
registry = ModelRegistry()
registry.load_all()
saved = load_active()
if saved and saved in registry.names():
    registry.set_active(saved)

DEV_TEST = os.environ.get("DEV_TEST", "1") == "1"  # default dev mode on

class SwitchReq(BaseModel):
    model_name: str

@app.get("/models")
def list_models():
    return {"available": registry.names(), "active": registry.active}

@app.post("/admin/switch")
def admin_switch(req: SwitchReq, _=Depends(require_admin_key)):
    name = req.model_name
    registry.set_active(name)
    save_active(name)
    return {"status":"ok","active":name}

@app.get("/score")
def score(url: str):
    # dev shortcuts
    if DEV_TEST:
        low = url.lower()
        if "/adult" in low or "adult" in low or "nsfw" in low:
            return {"url": url, "score": 0.95, "verdict": "block", "category": "adult", "used_model": "dev-rule"}
        if "/phish" in low or "phish" in low or "login" in low:
            return {"url": url, "score": 0.86, "verdict": "warn", "category": "phishing", "used_model": "dev-rule"}
        return {"url": url, "score": 0.05, "verdict": "allow", "category": "benign", "used_model": "dev-rule"}
    # production: use registry
    score_val, used = registry.score_url(url)
    verdict = "allow"
    if score_val > 0.75:
        verdict = "block"
    elif score_val > 0.4:
        verdict = "warn"
    return {"url": url, "score": score_val, "verdict": verdict, "category": "phishing", "used_model": used}
