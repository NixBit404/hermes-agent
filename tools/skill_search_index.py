"""BM25F retrieval over installed skills (search-first retrieval v2).

Mirrors tools/tool_search_catalog.py's tokenizer/BM25 (copied, not imported — that module
is a plugin-compat surface we deliberately don't touch) and extends it to per-field
weighted scoring over snapshot-v3 entries."""

from __future__ import annotations

import difflib
import functools
import math
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

import snowballstemmer

K1, B = 1.5, 0.75                       # matches the tool catalog
RESULT_DESC_LIMIT = 200
SKILL_SEARCH_FIELD_WEIGHTS: Dict[str, float] = {
    "name": 3.0, "description": 2.0, "category": 2.0, "triggers": 3.0, "meta": 1.5, "body": 0.5,
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_thread_local = threading.local()


@functools.lru_cache(maxsize=16384)
def _stem(token: str) -> str:
    """Stem one token, memoized across index rebuilds (tool_search_catalog.py's
    pattern: Snowball stemmers carry mutable parsing state and tool calls are
    dispatched on parallel threads, so the stemmer is one-per-thread, lazily
    created)."""
    if getattr(_thread_local, "stemmer", None) is None:
        _thread_local.stemmer = snowballstemmer.stemmer("english")
    return _thread_local.stemmer.stemWord(token)


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [_stem(token) for token in _TOKEN_RE.findall(text.lower())]


def _name_words(name: str) -> str:
    return re.sub(r"[_.:/\-]+", " ", name)


@dataclass
class SkillDoc:
    name: str
    category: str
    description: str
    fields: Dict[str, List[str]]
    dlw: float  # weighted length: sum(w_f * |tokens_f|)


def build_skill_docs(entries: Iterable[Dict[str, Any]]) -> List[SkillDoc]:
    docs: List[SkillDoc] = []
    for e in entries:
        name = str(e.get("frontmatter_name") or e.get("skill_name") or "").strip()
        if not name:
            continue  # fail-open parse artifacts
        sf = e.get("search_fields") or {}
        description = str(e.get("description") or "")
        fields = {
            "name": _tokenize(_name_words(name)),
            "description": _tokenize(description),
            "category": _tokenize(str(e.get("category") or "general").replace("/", " ")),
            "triggers": _tokenize(" ".join(sf.get("triggers") or [])),
            "meta": _tokenize(" ".join(list(sf.get("tags") or []) + list(sf.get("related_skills") or []))),
            "body": _tokenize(str(sf.get("body_head") or "") + " " + " ".join(sf.get("body_headings") or [])),
        }
        dlw = sum(SKILL_SEARCH_FIELD_WEIGHTS.get(f, 1.0) * len(t) for f, t in fields.items())
        docs.append(SkillDoc(name=name, category=str(e.get("category") or "general"),
                             description=description, fields=fields, dlw=dlw))
    return docs


class SkillSearchIndex:
    def __init__(self, docs: List[SkillDoc]):
        self.docs = docs
        self._by_name = {d.name.lower(): d for d in docs}
        self.avg_dlw = sum(d.dlw for d in docs) / max(len(docs), 1)

    def _wtf(self, doc: SkillDoc, token: str) -> float:
        return sum(w * doc.fields[f].count(token) for f, w in SKILL_SEARCH_FIELD_WEIGHTS.items() if doc.fields.get(f))

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not self.docs or limit <= 0:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        exact = self._by_name.get(query.strip().lower())
        n = len(self.docs)
        # DF per UNIQUE query token, computed once — the per-doc loop must not
        # recompute it (O(docs²·tokens): ~500-780ms on the real 428-skill library;
        # SPEC p95 target <25ms). Same summation as before, so scores are identical.
        df_map: Dict[str, int] = {q: sum(1 for d in self.docs if self._wtf(d, q) > 0)
                                  for q in set(q_tokens)}
        scored = []
        for doc in self.docs:
            s = float("inf") if doc is exact else self._score(q_tokens, doc, n, df_map)
            if s > 0:
                scored.append((s, doc))
        if not scored:  # substring fallback (tool-catalog precedent)
            ql = query.strip().lower()
            scored = [(0.1, d) for d in self.docs if ql in d.name.lower()]
        scored.sort(key=lambda x: (-x[0], x[1].name))
        return [{"name": d.name, "category": d.category,
                 "description": " ".join((d.description or "").split())[:RESULT_DESC_LIMIT],
                 "score": round(s, 3)} for s, d in scored[:limit]]

    def _score(self, q_tokens: List[str], doc: SkillDoc, n: int, df_map: Dict[str, int]) -> float:
        """BM25F over the doc's weighted fields; ``df_map`` carries per-token document
        frequencies precomputed by ``search`` (one pass, not one per doc·token)."""
        total = 0.0
        for q in q_tokens:
            tf = self._wtf(doc, q)
            if tf <= 0:
                continue
            df = df_map[q]
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            total += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * doc.dlw / max(self.avg_dlw, 1.0)))
        return total


def difflib_suggest(names: List[str], query: str, limit: int = 3) -> List[str]:
    """Closest names for a typo'd query, in ORIGINAL case: consumers match the
    returned name case-sensitively (skill_view retry) and look descriptions up
    by exact name, so lowercasing the matches here broke both."""
    lowered = {n.lower(): n for n in names}
    return [lowered[m] for m in difflib.get_close_matches(query.strip().lower(), list(lowered), n=limit, cutoff=0.6)]


def _recent(iso: Optional[str], days: int = 14) -> float:
    if not iso:
        return 0.0
    try:
        dt = datetime.fromisoformat(str(iso))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return 1.0 if dt >= datetime.now(timezone.utc) - timedelta(days=days) else 0.0
    except ValueError:
        return 0.0


def apply_usage_prior(hits: List[Dict[str, Any]], usage: Dict[str, Any],
                      excluded: frozenset) -> List[Dict[str, Any]]:
    """score += 0.5*ln(1+use_count) + 0.3*used_within_14d + 2.0*pinned; excluded: prior=0."""
    out = []
    for h in hits:
        rec = usage.get(h["name"]) or {}
        prior = 0.0 if h["name"] in excluded else (
            0.5 * math.log(1 + int(rec.get("use_count") or 0))
            + 0.3 * _recent(rec.get("last_used_at"))
            + (2.0 if rec.get("pinned") else 0.0))
        use_count = int(rec.get("use_count") or 0)
        h = dict(h, score=round(h["score"] + prior, 3),
                 badges=[b for b, on in (("pinned", rec.get("pinned")),
                                         (f"{use_count} uses", use_count > 0))
                         if on])
        out.append(h)
    out.sort(key=lambda h: (-h["score"], h["name"]))
    return out
