"""Writers for the two submission files.

Guarantees (checked before writing, independently of the official validator):
* exactly one row per Source-1 entity of the split, in input order;
* empty list for entities with no candidates / no matches;
* only S2-/S3- IDs, no duplicates within a list;
* every matched ID is also a candidate of the same entity.
ID lists are ordered by descending score so the files are human-inspectable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .ids import decode_ids, source_of


def _id_lists(s1_codes: np.ndarray, pairs: pd.DataFrame, score_col: str | None) -> pd.Series:
    if pairs.empty:
        return pd.Series("", index=pd.Index(s1_codes), dtype=str)
    p = pairs.drop_duplicates(["s1", "cand"])
    if score_col is not None:
        p = p.sort_values(["s1", score_col], ascending=[True, False], kind="stable")
    p = p.assign(cid=decode_ids(p["cand"].to_numpy()).to_numpy())
    lists = p.groupby("s1", sort=False)["cid"].agg(",".join)
    return lists.reindex(s1_codes, fill_value="")


def check_pairs(s1_codes: np.ndarray, cands: pd.DataFrame, matches: pd.DataFrame) -> None:
    s1_set = pd.Index(s1_codes)
    if not s1_set.is_unique:
        raise ValueError("duplicate S1 entities")
    for name, p in (("candidates", cands), ("matches", matches)):
        if len(p) and not np.isin(source_of(p["cand"].to_numpy()), (2, 3)).all():
            raise ValueError(f"{name} contain non S2/S3 ids")
        if len(p) and not p["s1"].isin(s1_set).all():
            raise ValueError(f"{name} reference unknown S1 ids")
    if len(matches):
        key = pd.MultiIndex.from_frame(matches[["s1", "cand"]])
        if not key.isin(pd.MultiIndex.from_frame(cands[["s1", "cand"]])).all():
            raise ValueError("a matched id is not among the entity's candidates")


def write_submission(out_dir: Path, s1_codes: np.ndarray, cands: pd.DataFrame, matches: pd.DataFrame,
                     score_col: str | None = "score") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    s1_codes = np.asarray(s1_codes, dtype=np.int64)
    check_pairs(s1_codes, cands, matches)
    s1_ids = decode_ids(s1_codes)
    stats = {}
    for fname, header, pairs in (
        ("candidate_pairs.tsv", "candidate_entity_ids", cands),
        ("matching_results.tsv", "matched_entity_ids", matches),
    ):
        col = score_col if score_col in pairs.columns else None
        lists = _id_lists(s1_codes, pairs, col)
        df = pd.DataFrame({"source1_entity_id": s1_ids.to_numpy(), header: lists.to_numpy()})
        path = out_dir / fname
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"source1_entity_id\t{header}\n")
            for a, b in zip(df["source1_entity_id"], df[header]):
                f.write(f"{a}\t{b}\n")
        stats[fname] = {"rows": int(len(df)), "empty_rows": int((df[header] == "").sum()), "pairs": int(len(pairs))}
    return stats


def read_id_list_file(path: Path) -> pd.DataFrame:
    """Read a matching/candidate TSV back into (s1, cand) int64 code pairs."""
    from .ids import encode_ids

    df = pd.read_csv(path, sep="\t", dtype=str, na_filter=False, quoting=3)
    col = df.columns[1]
    s = df.set_index("source1_entity_id")[col]
    s = s[s != ""].str.split(",").explode()
    if s.empty:
        return pd.DataFrame({"s1": pd.Series(dtype=np.int64), "cand": pd.Series(dtype=np.int64)})
    return pd.DataFrame({"s1": encode_ids(pd.Series(s.index)), "cand": encode_ids(s.reset_index(drop=True))})
