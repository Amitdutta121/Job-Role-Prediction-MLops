import os
import yaml
import joblib
import json
import time
from pathlib import Path
from typing import Dict


# -------------------------------
# Config Handling
# -------------------------------
def load_yaml(path: str) -> Dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)



def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def from_root(*parts):
    """Build an absolute path from project root."""
    return os.path.join(get_project_root(), *parts)


# -------------------------------
# Timer Decorator
# -------------------------------
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"[⏱️] {func.__name__} executed in {time.time() - start:.2f}s")
        return result
    return wrapper

# -------------------------------
# Save / Load Model
# -------------------------------
def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path: str):
    return joblib.load(path)



# -------------------------------
# Save / Load JSON
# -------------------------------
def save_json(data: Dict, path: str):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)

# -------------------------------
# Path Helper
# -------------------------------
def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]