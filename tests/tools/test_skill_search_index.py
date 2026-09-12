"""Tests for tools/skill_search_index.py — BM25F skill-retrieval core (retrieval v2, M1).

Contract per the task brief: sibling disambiguation (full description + triggers
beat title-only matches), trigger-field weighting, exact-name-first, substring
fallback, capped result descriptions, determinism, empty-corpus safety, and the
usage-prior formula 0.5*ln(1+use_count) + 0.3*recent + 2.0*pinned (exclusion
zeroes the prior only, the hit stays listed).
"""

from datetime import datetime, timedelta, timezone

CANARIES = [
    {"frontmatter_name": "live-macbook-search", "category": "devops",
     "description": "Searches the live MacBook desktop with Spotlight and mdfind; finds files, apps, and settings on the Mac.",
     "search_fields": {"triggers": ["search the web", "find files on the mac"], "tags": ["macbook", "search"], "related_skills": [], "body_headings": ["Workflow"], "body_head": "mdfind and Spotlight routing for the MacBook."}},
    {"frontmatter_name": "live-pi-52-search", "category": "devops",
     "description": "Routes search queries to the Pi52 box over SSH; use when the task must run on the Pi52 cluster.",
     "search_fields": {"triggers": [], "tags": ["pi52", "search"], "related_skills": [], "body_headings": [], "body_head": "SSH routing to Pi52."}},
    {"frontmatter_name": "3-tool-search", "category": "devops",
     "description": "Fallback three-tool search posture when no dedicated machine search applies.",
     "search_fields": {"triggers": [], "tags": [], "related_skills": [], "body_headings": [], "body_head": ""}},
    {"frontmatter_name": "pdf-forms", "category": "docs",
     "description": "Fill and flatten PDF forms with pypdf.",
     "search_fields": {"triggers": [], "tags": ["pdf"], "related_skills": [], "body_headings": [], "body_head": ""}},
]


class TestSkillSearchIndex:
    def _index(self, entries=None):
        from tools.skill_search_index import SkillSearchIndex, build_skill_docs
        return SkillSearchIndex(build_skill_docs(entries if entries is not None else CANARIES))

    def test_sibling_disambiguation_full_desc_wins_over_title(self):
        hits = self._index().search("find files on the mac with spotlight", limit=3)
        assert hits[0]["name"] == "live-macbook-search"

    def test_trigger_field_ranks_first(self):
        hits = self._index().search("search the web", limit=3)
        assert hits[0]["name"] == "live-macbook-search"

    def test_exact_name_match_ranks_first(self):
        hits = self._index().search("pdf-forms", limit=2)
        assert hits[0]["name"] == "pdf-forms"

    def test_multi_token_query_ranking_and_scores_pinned(self):
        """Correctness pin for the DF-map refactor of _score (df computed once per
        unique query token in search() instead of per doc·token): a known multi-token
        query must keep returning EXACTLY these ranked names and scores (captured from
        the pre-refactor implementation; the refactor must be bit-identical)."""
        hits = self._index().search("find files on the mac with spotlight", limit=3)
        assert [(h["name"], h["score"]) for h in hits] == [
            ("live-macbook-search", 11.367), ("live-pi-52-search", 2.275), ("pdf-forms", 1.182)]

    def test_substring_fallback_when_no_token_hits(self):
        hits = self._index().search("pi52", limit=2)   # "pi52" stems to itself; lives in name/tags
        assert any(h["name"] == "live-pi-52-search" for h in hits)

    def test_substring_fallback_fires_only_with_zero_token_hits(self):
        """Ordering rule 3: the fixed-low-score (0.1) name-substring fallback fires
        ONLY when NO query token matches any field. "c-de" tokenizes to c/de (both
        stem to themselves) which appear in no doc field — but the raw string is a
        substring of "abc-def", so the fallback returns exactly that doc at 0.1."""
        entry = {"frontmatter_name": "abc-def", "category": "docs",
                 "description": "Fill and flatten PDF forms with pypdf.",
                 "search_fields": {"triggers": [], "tags": [], "related_skills": [], "body_headings": [], "body_head": ""}}
        other = {"frontmatter_name": "zzz-qqq", "category": "docs",
                 "description": "Unrelated widget tooling.",
                 "search_fields": {"triggers": [], "tags": [], "related_skills": [], "body_headings": [], "body_head": ""}}
        hits = self._index([entry, other]).search("c-de", limit=5)
        assert [h["name"] for h in hits] == ["abc-def"]   # non-substring sibling absent
        assert hits[0]["score"] == 0.1

    def test_result_description_capped_and_shape(self):
        hits = self._index().search("search the web", limit=1)
        assert set(hits[0]) == {"name", "category", "description", "score"}
        assert len(hits[0]["description"]) <= 200

    def test_empty_and_garbage_queries_return_empty(self):
        assert self._index().search("") == []
        assert self._index().search("   ") == []
        assert self._index().search("!!!---") == []

    def test_deterministic(self):
        a = self._index().search("search", limit=4)
        b = self._index().search("search", limit=4)
        assert [h["name"] for h in a] == [h["name"] for h in b]

    def test_empty_entries_ok(self):
        assert self._index([]).search("anything") == []

    def test_difflib_suggest(self):
        from tools.skill_search_index import difflib_suggest
        names = [e["frontmatter_name"] for e in CANARIES]
        assert "live-macbook-search" in difflib_suggest(names, "live-macbook-serach", limit=2)

    def test_difflib_suggest_returns_original_case(self):
        """Suggestions must keep the corpus' casing: the skill_view retry is
        case-sensitive and the description lookup is keyed by the exact name, so a
        lowercased match would fail both."""
        from tools.skill_search_index import difflib_suggest
        assert difflib_suggest(["Mac-Search", "pdf-forms"], "mac-serach") == ["Mac-Search"]


class TestUsagePrior:
    def test_prior_monotonic_and_pinned_and_excluded(self):
        import math
        from tools.skill_search_index import apply_usage_prior
        # Used yesterday relative to *runtime* (hardcoding a date would flip the
        # +0.3 recency term to 0 once the calendar passes last_used_at + 14d).
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        hits = [{"name": "a", "score": 5.0}, {"name": "b", "score": 5.0}]
        usage = {"a": {"use_count": 9, "last_used_at": yesterday, "pinned": False},
                 "b": {"use_count": 0, "last_used_at": None, "pinned": True}}
        out = apply_usage_prior(hits, usage, excluded=frozenset())
        assert out[0]["score"] > out[1]["score"]                       # pinned beats use_count 9 (best-first)
        assert math.isclose(out[1]["score"], round(5.0 + 0.5 * math.log(10) + 0.3, 3), rel_tol=1e-9)  # +0.3 recency: used yesterday
        out_x = apply_usage_prior(hits, usage, excluded=frozenset({"b"}))
        assert out_x[1]["score"] == 5.0                                # excluded: prior zeroed, still listed
