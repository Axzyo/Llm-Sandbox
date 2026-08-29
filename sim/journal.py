import json
import os
import threading
import time


class Journal:
    def __init__(self, path: str, run_id: str):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8")
        self.run_id = run_id
        self.t0 = time.monotonic()
        self.lock = threading.Lock()

    def log(self, actor: str, type_: str, **payload) -> None:
        rec = {
            "t": round(time.monotonic() - self.t0, 3),
            "run": self.run_id,
            "actor": actor,
            "type": type_,
            "payload": payload,
        }
        line = json.dumps(rec) + "\n"
        with self.lock:
            self.fh.write(line)
            self.fh.flush()

    def close(self) -> None:
        with self.lock:
            self.fh.close()
