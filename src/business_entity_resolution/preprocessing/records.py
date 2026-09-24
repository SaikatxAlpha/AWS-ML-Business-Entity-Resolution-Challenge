"""Normalised record tables (all three sources of a split in one frame), with caching.

Columns: code (int64 id), source (int8), country, name_raw, addr_raw, name_pre, addr_pre,
nums (space-joined address numbers), and after ``normalized``: name_norm, name_core, addr_norm.

The label-dependent part (token maps + weak-token set) is a *normaliser* learned from a set
of training S1 entities only and saved to ``models/normalizer_<tag>.json``; for validation
experiments it is learned from the training folds, for the final model from all train data.
"""
from __future__ import annotations

import json
import time

import pandas as pd

from ..config import MODELS_DIR, cache_path
from ..data.loader import load_source
from ..utils.ids import encode_ids, source_of
from .address_normalization import address_numbers, normalize_addresses, prenormalize_addresses
from .name_normalization import instability_tokens, normalize_names, prenormalize_names, suffix_tokens
from .token_maps import apply_map, mine_token_maps

PREPROCESSING_VERSION = "p2"


def prenormalized(split: str) -> pd.DataFrame:
    p = cache_path("records", f"{split}_pre_{PREPROCESSING_VERSION}.parquet")
    if p.exists():
        return pd.read_parquet(p)
    t = time.time()
    parts = []
    for k in (1, 2, 3):
        d = load_source(split, k)
        parts.append(pd.DataFrame({
            "code": encode_ids(d["entity_id"]),
            "country": d["country"].to_numpy(),
            "name_raw": d["business_name"].to_numpy(),
            "addr_raw": d["business_address"].to_numpy(),
        }))
    df = pd.concat(parts, ignore_index=True)
    df["source"] = source_of(df["code"])
    df["name_pre"] = prenormalize_names(df["name_raw"])
    df["addr_pre"] = prenormalize_addresses(df["addr_raw"])
    df["nums"] = address_numbers(df["addr_raw"])
    df.to_parquet(p, index=False)
    print(f"[records] prenormalized {split}: {len(df):,} rows in {time.time() - t:.0f}s", flush=True)
    return df


def learn_normalizer(train_s1: pd.Series, truth: pd.DataFrame, tag: str) -> dict:
    """Mine token maps and weak tokens from positives of the given training S1 entities."""
    path = MODELS_DIR / f"normalizer_{tag}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rec = prenormalized("train").set_index("code")
    pos = truth[truth["s1"].isin(set(train_s1))]
    a = rec.loc[pos["s1"].to_numpy()]
    b = rec.loc[pos["cand"].to_numpy()]
    pairs = pd.DataFrame({
        "country": a["country"].to_numpy(),
        "a_name": a["name_pre"].to_numpy(), "b_name": b["name_pre"].to_numpy(),
        "a_addr": a["addr_pre"].to_numpy(), "b_addr": b["addr_pre"].to_numpy(),
    })
    s1 = rec[(rec["source"] == 1) & rec.index.isin(set(train_s1))]
    maps = mine_token_maps(pairs, pd.DataFrame({"country": s1["country"], "name": s1["name_pre"], "addr": s1["addr_pre"]}))
    weak = {}
    for country, grp in pairs.groupby("country"):
        m = maps[country]["name"]
        weak[country] = instability_tokens(
            pd.Series([apply_map(x, m) for x in grp["a_name"]]),
            pd.Series([apply_map(x, m) for x in grp["b_name"]]),
        )
    norm = {"tag": tag, "n_train_s1": int(len(set(train_s1))), "n_pos_pairs": int(len(pairs)),
            "maps": maps, "weak_tokens": weak, "preprocessing_version": PREPROCESSING_VERSION}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(norm, indent=1, ensure_ascii=False), encoding="utf-8")
    return norm


def normalized(split: str, normalizer: dict) -> pd.DataFrame:
    tag = normalizer["tag"]
    p = cache_path("records", f"{split}_norm_{PREPROCESSING_VERSION}_{tag}.parquet")
    if p.exists():
        return pd.read_parquet(p)
    t = time.time()
    df = prenormalized(split)
    name_norm = pd.Series("", index=df.index, dtype=str)
    name_core = pd.Series("", index=df.index, dtype=str)
    addr_norm = pd.Series("", index=df.index, dtype=str)
    weak_used = {}
    for country, idx in df.groupby("country").indices.items():
        maps = normalizer["maps"].get(country, {})          # unseen country -> no learned map
        rows = df.index[idx]
        nm = maps.get("name", {})
        s1_rows = rows[df.loc[rows, "source"].to_numpy() == 1]
        s1_norm = pd.Series([apply_map(x, nm) for x in df.loc[s1_rows, "name_pre"]])
        weak = set(normalizer["weak_tokens"].get(country, {})) | suffix_tokens(s1_norm)
        weak_used[country] = sorted(weak)
        n, c = normalize_names(df.loc[rows, "name_pre"], nm, weak)
        name_norm.loc[rows] = n.to_numpy()
        name_core.loc[rows] = c.to_numpy()
        addr_norm.loc[rows] = normalize_addresses(df.loc[rows, "addr_pre"], maps.get("addr", {})).to_numpy()
    df = df.assign(name_norm=name_norm, name_core=name_core, addr_norm=addr_norm)
    df.to_parquet(p, index=False)
    cache_path("records", f"{split}_weak_{tag}.json").write_text(json.dumps(weak_used, indent=1), encoding="utf-8")
    print(f"[records] normalized {split} ({tag}): {time.time() - t:.0f}s", flush=True)
    return df
