"""Leakage-safe validation design.

* Splits are made over **Source-1 entities** only. A pair's fold is the fold of its S1
  entity, so an S1 entity never contributes labels to both training and validation.
* Folds are stratified by ``country × match-cardinality bucket`` (0, 1, 2, ..., 6+), so
  singletons, one-match and multi-match entities are represented in every fold.
* Validation matching runs against the **entire** train S2/S3 pool (as test matching runs
  against the entire test S2/S3 pool); inference code never receives ground truth.
* ``country_holdout`` gives a train-on-one-country / validate-on-another split used as a
  proxy for the unseen test country.

Folds are deterministic given ``SEED`` and cached in ``cache/splits``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SEED, cache_path
from ..data.loader import load_ground_truth, load_source
from ..utils.ids import encode_ids

SPLIT_VERSION = "v1"
N_FOLDS = 5
VAL_FOLD = 0
CARD_CAP = 6


def truth_pairs() -> pd.DataFrame:
    """All positive train pairs as int64 codes: columns s1, cand."""
    p = cache_path("labels", "truth_pairs.parquet")
    if p.exists():
        return pd.read_parquet(p)
    gt = load_ground_truth()
    s = gt.set_index("source1_entity_id")["matched_entity_ids"]
    s = s[s != ""].str.split(",").explode()
    out = pd.DataFrame({"s1": encode_ids(pd.Series(s.index)), "cand": encode_ids(s.reset_index(drop=True))})
    out.to_parquet(p, index=False)
    return out


def folds() -> pd.DataFrame:
    """One row per train S1 entity: s1 (code), country, card, stratum, fold."""
    p = cache_path("splits", f"folds_{SPLIT_VERSION}.parquet")
    if p.exists():
        return pd.read_parquet(p)
    s1 = load_source("train", 1)[["entity_id", "country"]]
    gt = load_ground_truth()
    card = np.where(gt["matched_entity_ids"] == "", 0, gt["matched_entity_ids"].str.count(",") + 1)
    df = pd.DataFrame({"entity_id": gt["source1_entity_id"], "card": card}).merge(s1, on="entity_id", how="left")
    if df["country"].isna().any():
        raise ValueError("ground-truth S1 id missing from train_source1")
    df["s1"] = encode_ids(df["entity_id"])
    df = df.sort_values("s1", kind="stable").reset_index(drop=True)
    df["stratum"] = df["country"].astype(str) + "|" + np.minimum(df["card"], CARD_CAP).astype(str)
    rng = np.random.default_rng(SEED)
    df["fold"] = -1
    for _, idx in df.groupby("stratum").indices.items():
        perm = rng.permutation(len(idx))
        df.loc[df.index[idx[perm]], "fold"] = np.arange(len(idx)) % N_FOLDS
    out = df[["s1", "country", "card", "stratum", "fold"]]
    out.to_parquet(p, index=False)
    return out


def split(fold: int = VAL_FOLD) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = folds()
    return f[f["fold"] != fold].reset_index(drop=True), f[f["fold"] == fold].reset_index(drop=True)


def country_holdout(val_country: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = folds()
    return f[f["country"] != val_country].reset_index(drop=True), f[f["country"] == val_country].reset_index(drop=True)


def stratified_sample(entities: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Proportional stratified subsample (for fast development runs)."""
    if n >= len(entities):
        return entities
    frac = n / len(entities)
    parts = [g.sample(frac=frac, random_state=seed) for _, g in entities.groupby("stratum")]
    return pd.concat(parts).sort_values("s1").reset_index(drop=True)


def cardinality_bucket(card: pd.Series) -> pd.Series:
    return np.minimum(card, CARD_CAP).astype(str).replace(str(CARD_CAP), f"{CARD_CAP}+")
