"""Address normalisation.

Stage A (label-free): remove apostrophes, NFKC + transliteration + lowercase + punctuation
-> space, leading zeros stripped from numeric tokens, literal ``null`` tokens removed.
Stage B: country-conditional token map learned from training positives (``rd``->``road``,
``mh``/``mharastr``->``maharashtra``, ``tx``->``texas`` ...).

Numbers are extracted separately (all digit runs, leading zeros removed) because they are
the strongest single matching signal (see reports/dataset_profile.md, observation 8).
"""
from __future__ import annotations

import pandas as pd

from .normalization import basic_normalize_series
from .token_maps import apply_map
from .tokenization import numbers

DROP_TOKENS = frozenset({"null"})


def _clean_tokens(text: str) -> str:
    out = []
    for t in text.split():
        if t in DROP_TOKENS:
            continue
        if t.isdigit():
            t = t.lstrip("0") or "0"
        out.append(t)
    return " ".join(out)


def prenormalize_addresses(addrs: pd.Series) -> pd.Series:
    s = addrs.astype(str).str.replace("'", "", regex=False).str.replace("’", "", regex=False)
    s = basic_normalize_series(s)
    return pd.Series([_clean_tokens(t) for t in s], index=addrs.index, dtype=str)


def normalize_addresses(pre: pd.Series, mapping: dict[str, str]) -> pd.Series:
    return pd.Series([apply_map(t, mapping) for t in pre], index=pre.index, dtype=str)


def address_numbers(raw_addrs: pd.Series) -> pd.Series:
    """Space-joined digit runs of the raw address (leading zeros removed)."""
    return pd.Series([" ".join(numbers(a)) for a in raw_addrs.astype(str)], index=raw_addrs.index, dtype=str)
