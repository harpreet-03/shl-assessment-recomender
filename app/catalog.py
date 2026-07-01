"""
catalog.py
----------
Loads app/data/catalog.json and exposes a BM25-based retrieval index.

Design choice: BM25 (rank_bm25) instead of embeddings.
- No API key / network call needed at retrieval time -> fast, free, deterministic.
- The catalog is short, structured text (name + description + test-type labels),
  which is exactly the regime where lexical retrieval (BM25) is competitive with
  embeddings and a lot cheaper/faster to run inside an 8-turn / 30s-timeout budget.
- It also makes the closed-set guarantee easy to enforce: the LLM is only ever
  shown and allowed to pick from a pre-retrieved candidate list of *real* catalog
  rows -> it structurally cannot hallucinate a name/URL that isn't in catalog.json.
"""
import json
import os
import re
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

TEST_TYPE_LEGEND = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

TOKEN_RE = re.compile(r"[a-z0-9\+\#\.]+")


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


class Catalog:
    def __init__(self, path: str):
        self.path = path
        self.items: List[Dict[str, Any]] = []
        self._bm25: BM25Okapi = None
        self._corpus_tokens: List[List[str]] = []
        self.load()

    def load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        for item in self.items:
            item.setdefault("test_type", [])
            item.setdefault("description", "")
            item.setdefault("job_levels", [])
            item.setdefault("languages", [])
            labels = [TEST_TYPE_LEGEND.get(t, t) for t in item["test_type"]]
            item["test_type_labels"] = labels
            doc = " ".join([
                item["name"], item["name"], item["description"],
                " ".join(labels), " ".join(item.get("job_levels", [])),
            ])
            self._corpus_tokens.append(_tokenize(doc))
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def __len__(self):
        return len(self.items)

    def find_by_name(self, name: str) -> Dict[str, Any] | None:
        name_l = name.strip().lower()
        for item in self.items:
            if item["name"].strip().lower() == name_l:
                return item
        # loose contains-match fallback
        for item in self.items:
            if name_l in item["name"].strip().lower() or item["name"].strip().lower() in name_l:
                return item
        return None

    def find_by_url(self, url: str) -> Dict[str, Any] | None:
        for item in self.items:
            if item["url"] == url:
                return item
        return None

    def search(self, query: str, top_k: int = 15, test_type_filter: List[str] = None) -> List[Dict[str, Any]]:
        """BM25 search over the catalog. Optionally restrict to items that
        contain at least one of the requested test_type codes."""
        if not self._bm25 or not query.strip():
            candidates = self.items
        else:
            tokens = _tokenize(query)
            scores = self._bm25.get_scores(tokens)
            ranked = sorted(range(len(self.items)), key=lambda i: scores[i], reverse=True)
            candidates = [self.items[i] for i in ranked]

        if test_type_filter:
            wanted = set(t.upper() for t in test_type_filter)
            candidates = [c for c in candidates if wanted.intersection(set(c.get("test_type", [])))] or candidates

        return candidates[:top_k]


_CATALOG_SINGLETON: Catalog = None


def get_catalog() -> Catalog:
    global _CATALOG_SINGLETON
    if _CATALOG_SINGLETON is None:
        path = os.environ.get("CATALOG_PATH", os.path.join(os.path.dirname(__file__), "data", "catalog.json"))
        _CATALOG_SINGLETON = Catalog(path)
    return _CATALOG_SINGLETON
