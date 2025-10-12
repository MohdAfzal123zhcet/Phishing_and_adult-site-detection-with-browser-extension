# model_service/persistence.py
import json, os
FILE = "model_service/active_model.json"
def save_active(name):
    with open(FILE,"w") as f: json.dump({"active":name}, f)
def load_active():
    if os.path.exists(FILE):
        try:
            return json.load(open(FILE))['active']
        except: return None
    return None
