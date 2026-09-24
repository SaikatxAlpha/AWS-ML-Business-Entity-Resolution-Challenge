"""Loading of raw TSV files with Parquet caching.

Raw files are never modified. The first load of each file parses the TSV and writes a
Parquet copy under ``cache/raw``; later loads read the Parquet copy (much faster). The
cache is invalidated when the source TSV's size or mtime changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import cache_path, ground_truth_path, source_path
from .schema import GT_SCHEMA, READ_KWARGS, SOURCE_SCHEMA


def _fingerprint(path: Path) -> dict:
    st = path.stat()
    return {"path": str(path), "size": st.st_size, "mtime": int(st.st_mtime)}


def _cached_read(tsv: Path, name: str, expected_cols: list[str], refresh: bool) -> pd.DataFrame:
    pq = cache_path("raw", f"{name}.parquet")
    meta = pq.with_suffix(".json")
    fp = _fingerprint(tsv)
    if not refresh and pq.exists() and meta.exists() and json.loads(meta.read_text()) == fp:
        return pd.read_parquet(pq)
    df = pd.read_csv(tsv, **READ_KWARGS)
    if list(df.columns) != expected_cols:
        raise ValueError(f"{tsv}: unexpected columns {list(df.columns)}; expected {expected_cols}")
    df.to_parquet(pq, index=False)
    meta.write_text(json.dumps(fp))
    return df


def load_source(split: str, source: int, refresh: bool = False) -> pd.DataFrame:
    """Load one source file as a DataFrame of strings."""
    return _cached_read(source_path(split, source), f"{split}_source{source}", SOURCE_SCHEMA, refresh)


def load_ground_truth(refresh: bool = False) -> pd.DataFrame:
    """Raw ground truth: one row per S1 entity, comma-separated match list (may be empty)."""
    return _cached_read(ground_truth_path(), "train_ground_truth", GT_SCHEMA, refresh)


def ground_truth_pairs(gt: pd.DataFrame | None = None) -> pd.DataFrame:
    """Explode the ground truth to one row per positive (s1_id, match_id) pair.

    Singletons produce no rows; use :func:`load_ground_truth` to enumerate all S1 entities.
    """
    gt = load_ground_truth() if gt is None else gt
    s = gt.set_index("source1_entity_id")["matched_entity_ids"]
    s = s[s != ""].str.split(",").explode()
    out = s.reset_index()
    out.columns = ["s1_id", "match_id"]
    return out


def load_all_sources(split: str) -> dict[int, pd.DataFrame]:
    return {k: load_source(split, k) for k in (1, 2, 3)}
