"""Business-name normalisation.

Stage A (label-free, used for mining too):
    remove apostrophes, ``&`` -> ``and``, NFKC + transliteration + lowercase + punctuation
    -> space, OCR digit-for-letter repair, join runs of single letters (``l l c`` -> ``llc``).
Stage B (country-conditional learned map):
    apply the token equivalence map mined from training positives.
Core name:
    remove *weak* tokens — legal forms and generic descriptors — so matching focuses on the
    distinctive part of the name. Weak tokens come from two data-driven sources:
      1. label-mined instability: tokens that frequently appear on only one side of a true pair
         (``llc``, ``limited``, ``private``, ``center``, ...), per training country;
      2. label-free suffix detection on the split's own Source-1 names: tokens that are
         frequently the last token of a name and rarely appear elsewhere (catches unseen
         countries' legal forms such as ``sarl`` / ``sas`` without any labels).
The full normalised name is always kept alongside the core.
"""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from .normalization import basic_normalize_series
from .token_maps import apply_map
from .tokenization import fix_ocr_token

STOPWORDS = frozenset({"the", "and", "of"})
_SINGLE_RUN = re.compile(r"\b(?:[a-z] ){1,}[a-z]\b")


def _join_single_letters(text: str) -> str:
    return _SINGLE_RUN.sub(lambda m: m.group(0).replace(" ", ""), text)


def prenormalize_names(names: pd.Series) -> pd.Series:
    s = names.astype(str).str.replace("'", "", regex=False).str.replace("’", "", regex=False)
    s = s.str.replace("&", " and ", regex=False)
    s = basic_normalize_series(s)
    out = []
    for text in s:
        toks = [fix_ocr_token(t) for t in text.split()]
        out.append(_join_single_letters(" ".join(toks)))
    return pd.Series(out, index=names.index, dtype=str)


def suffix_tokens(names_norm: pd.Series, min_share: float = 0.003, min_last_ratio: float = 0.6) -> set[str]:
    """Label-free legal-form detection over one country's Source-1 names."""
    last = Counter()
    anywhere = Counter()
    for text in names_norm:
        toks = text.split()
        if not toks:
            continue
        last[toks[-1]] += 1
        anywhere.update(set(toks))
    n = max(1, len(names_norm))
    return {t for t, c in last.items() if c / n >= min_share and c / anywhere[t] >= min_last_ratio}


def instability_tokens(a_names: pd.Series, b_names: pd.Series, min_share: float = 0.002, min_rate: float = 0.5) -> dict[str, float]:
    """Common tokens that often appear on only one side of true pairs (label-mined, training folds only).

    ``min_share`` (fraction of pairs containing the token) restricts the set to *generic*
    tokens: rare transliterated words (``helthkeyr``) are also one-sided, but they are
    informative and must stay in the core name.
    """
    one_side, either = Counter(), Counter()
    n = 0
    for x, y in zip(a_names, b_names):
        sx, sy = set(x.split()), set(y.split())
        either.update(sx | sy)
        one_side.update(sx ^ sy)
        n += 1
    min_pairs = max(1, int(min_share * n))
    return {t: round(one_side[t] / c, 4) for t, c in either.items() if c >= min_pairs and one_side[t] / c >= min_rate}


def core_name(name_norm: str, weak: set[str]) -> str:
    toks = [t for t in name_norm.split() if t not in weak and t not in STOPWORDS]
    return " ".join(toks) if toks else name_norm


def normalize_names(pre: pd.Series, mapping: dict[str, str], weak: set[str]) -> tuple[pd.Series, pd.Series]:
    norm = [apply_map(t, mapping) for t in pre]
    core = [core_name(t, weak) for t in norm]
    return pd.Series(norm, index=pre.index, dtype=str), pd.Series(core, index=pre.index, dtype=str)
