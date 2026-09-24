"""Token equivalence maps learned from training positives (no external dictionaries).

For every true (S1, S2/S3) pair whose token sets differ by exactly one token on each side,
the substitution (match token -> S1 token) is counted, per country and field. A
substitution is accepted when

* it occurs at least ``min_count`` times,
* it is *consistent*: of all substitutions in which the match token takes part, at least
  ``min_consistency`` map it to the same S1 token (rejects generic insertions such as
  ``private -> center``), and
* it is *string-plausible*: abbreviation-compatible (same first letter, in-order
  subsequence: ``rd``/``road``, ``mh``/``maharashtra``, ``tx``/``texas``) or
  character similarity >= ``min_ratio`` (``praivet``/``private``, ``lnc``/``inc``). This
  rejects coincidences such as ``shri -> limited``.

Accepted pairs are merged with union-find; each class is represented by its most frequent
member in Source 1 (the clean canonical source). Maps are keyed by the country labels
present in the training labels, so an unseen country simply gets no learned map.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from .tokenization import abbreviation_compatible

FIELDS = ("name", "addr")


class _UF:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        self.p[self.find(a)] = self.find(b)


def count_substitutions(a_texts, b_texts) -> Counter:
    subs: Counter = Counter()
    for x, y in zip(a_texts, b_texts):
        sx, sy = set(x.split()), set(y.split())
        dx, dy = sx - sy, sy - sx
        if len(dx) == 1 and len(dy) == 1:
            subs[(next(iter(dy)), next(iter(dx)))] += 1  # (match token, S1 token)
    return subs


def accept_substitutions(subs: Counter, min_count: int, min_consistency: float, min_ratio: float) -> list[tuple]:
    by_match: dict[str, int] = defaultdict(int)
    for (m, _), c in subs.items():
        by_match[m] += c
    out = []
    for (m, s), c in subs.items():
        if c < min_count or m.isdigit() or s.isdigit():
            continue
        if c / by_match[m] < min_consistency:
            continue
        if abbreviation_compatible(m, s) or fuzz.ratio(m, s) >= min_ratio:
            out.append((m, s, c))
    return out


def build_map(pairs: list[tuple], s1_token_freq: Counter) -> dict[str, str]:
    uf = _UF()
    for m, s, _ in pairs:
        uf.union(m, s)
    groups: dict[str, list[str]] = defaultdict(list)
    for tok in list(uf.p):
        groups[uf.find(tok)].append(tok)
    mapping = {}
    for members in groups.values():
        canon = max(members, key=lambda t: (s1_token_freq.get(t, 0), len(t), t))
        for t in members:
            if t != canon:
                mapping[t] = canon
    return mapping


def mine_token_maps(pos: pd.DataFrame, s1_texts: pd.DataFrame, min_count: int = 25,
                    min_consistency: float = 0.5, min_ratio: float = 70.0) -> dict:
    """``pos``: positive pairs with columns country, a_name, b_name, a_addr, b_addr (pre-normalised text).
    ``s1_texts``: S1 records with columns country, name, addr (for canonical-form frequencies)."""
    maps: dict = {"params": {"min_count": min_count, "min_consistency": min_consistency, "min_ratio": min_ratio}}
    for country, grp in pos.groupby("country"):
        s1c = s1_texts[s1_texts["country"] == country]
        maps[country] = {}
        for field in FIELDS:
            subs = count_substitutions(grp[f"a_{field}"], grp[f"b_{field}"])
            acc = accept_substitutions(subs, min_count, min_consistency, min_ratio)
            freq = Counter(t for x in s1c[field] for t in x.split())
            maps[country][field] = build_map(acc, freq)
            maps[country][f"{field}_evidence"] = sorted(([m, s, c] for m, s, c in acc), key=lambda r: -r[2])
    return maps


def save_maps(maps: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(maps, indent=1, ensure_ascii=False), encoding="utf-8")


def load_maps(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_map(text: str, mapping: dict[str, str]) -> str:
    if not mapping:
        return text
    return " ".join(mapping.get(t, t) for t in text.split())
