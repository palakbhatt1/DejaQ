import json
import os
import threading
import uuid

import numpy as np

from .embeddings import DIM, embed
from .normalize import normalize


class Store:
    """In-memory question collection with a vector matrix and an exact-match index.

    Pass `path` to persist to JSON; omit it for an in-memory store (tests, eval).
    Tags are free-form and never used for duplicate logic.
    """

    def __init__(self, path: str | None = None):
        self.path = path
        self.items: list[dict] = []                      # {id, text, norm, tags}
        self.vecs = np.zeros((0, DIM), dtype="float32")
        self.norms: dict[str, int] = {}                  # normalized text -> index
        self.lock = threading.RLock()
        if path and os.path.exists(path):
            self._load()

    # ---- write ----
    def add(self, text: str, tags: dict | None = None, persist: bool = True) -> dict:
        n, v = normalize(text), embed([text])
        with self.lock:
            self.items.append({"id": uuid.uuid4().hex[:8], "text": text, "norm": n, "tags": tags or {}})
            self.vecs = np.vstack([self.vecs, v])
            self.norms.setdefault(n, len(self.items) - 1)
            if persist and self.path:
                self._save()
            return self.items[-1]

    def delete(self, qid: str, persist: bool = True) -> bool:
        with self.lock:
            idx = next((i for i, item in enumerate(self.items) if item["id"] == qid), None)
            if idx is None:
                return False
            del self.items[idx]
            self.vecs = np.delete(self.vecs, idx, axis=0)
            self.norms = {}
            for j, item in enumerate(self.items):
                self.norms.setdefault(item["norm"], j)
            if persist and self.path:
                self._save()
            return True

    # ---- read ----
    def exact(self, norm: str) -> dict | None:
        with self.lock:
            i = self.norms.get(norm)
            return self.items[i] if i is not None else None

    def search(self, text: str, k: int = 5) -> list[tuple[dict, float]]:
        q = embed([text])[0]
        with self.lock:
            if not self.items:
                return []
            s = self.vecs @ q
            idx = np.argsort(-s)[:k]
            return [(self.items[i], float(s[i])) for i in idx]

    def coverage(self, phrases: list[str]) -> list[float]:
        """Max similarity of each phrase to any stored question (0 if the store is empty)."""
        if not phrases:
            return []
        q = embed(phrases)
        with self.lock:
            if not self.items:
                return [0.0] * len(phrases)
            return [float(x) for x in (q @ self.vecs.T).max(axis=1)]

    def diverse_sample(self, k: int = 10) -> list[dict]:
        """Farthest-point sampling: spreads the sample across the whole collection."""
        with self.lock:
            n = len(self.items)
            if n <= k:
                return list(self.items)
            chosen = [0]
            d = 1 - self.vecs @ self.vecs[0]
            while len(chosen) < k:
                i = int(np.argmax(d))
                chosen.append(i)
                d = np.minimum(d, 1 - self.vecs @ self.vecs[i])
            return [self.items[i] for i in chosen]

    def filter(self, tag_key: str | None = None, tag_value: str | None = None) -> list[dict]:
        with self.lock:
            out = list(self.items)
        if tag_key is not None:
            out = [i for i in out if tag_key in i["tags"]
                   and (tag_value is None or str(i["tags"][tag_key]) == tag_value)]
        return out

    # ---- persistence ----
    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.items, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    def _load(self):
        with open(self.path, encoding="utf-8") as f:
            self.items = json.load(f)
        if self.items:
            self.vecs = embed([i["text"] for i in self.items])
        for j, i in enumerate(self.items):
            self.norms.setdefault(i["norm"], j)
