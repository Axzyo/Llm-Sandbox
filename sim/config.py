"""config.json loading, shared by the pygame app and the headless episode runner."""
import json
import os

DEFAULTS = {
    "ollama_url": "http://localhost:11434",
    "model": "gemma4",
    "temperature": 0.2,
    "num_predict": 400,
    "keep_alive": "30m",
    "memory_k": 5,
    "memory_halflife_s": 300.0,
    "interact_range": 4,
}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            cfg.update(json.load(fh))
    return cfg
