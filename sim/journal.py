import json
import os
import threading


class Journal:
    def __init__(self, path: str, run_id: str):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8")
        self.run_id = run_id
        self.lock = threading.Lock()
        # every record is stamped with sim time. Sim time is 0 until an Engine
        # exists (it starts its clock there and re-points this at itself).
        self.clock = lambda: 0.0

    def log(self, actor: str, type_: str, **payload) -> None:
        rec = {
            "t": round(self.clock(), 3),
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
