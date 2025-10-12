# model_service/auth.py
from fastapi import Header, HTTPException
import os
ADMIN_KEY = os.environ.get("ADMIN_API_KEY", "please-change")
def require_admin_key(x_api_key: str = Header(None)):
    if x_api_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    return True
