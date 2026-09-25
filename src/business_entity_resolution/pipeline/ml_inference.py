"""Learned-matcher inference (``models/matcher_spec.json`` from ``pipeline.train_pipeline``).

RAW TSVs -> normalisation (validated normaliser) -> per-country index (split's own IDF)
-> multi-block candidates (validated per-block K) -> candidate_pairs -> pair features
-> matcher probability -> (calibration) -> decision policy / abstention -> matching_results
-> official validator.

Each country is processed independently (open set of labels) and scored in batches of
complete S1 candidate lists, so memory stays bounded on the full test set.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ..blocking.candidate_generator import generate_candidates
from ..blocking.index import CountryIndex
from ..config import MODELS_DIR, OUTPUT_DIR, cache_path
from ..decision.abstention import context_matrix, policy_mask
from ..evaluation.validation import folds, truth_pairs
from ..features.feature_builder import build_features
from ..preprocessing.records import INDEX_COLUMNS, countries, learn_normalizer, normalized, normalized_country
from ..utils.io import write_submission
from .inference_pipeline import run_validator

ALL_BLOCKS = ["fielded", "addr", "name_char", "name_word"]


def load_matcher(spec: dict):
    import lightgbm as lgb
    import xgboost as xgb

    if spec["model"] == "xgb":
        bst = xgb.Booster()
        bst.load_model(str(MODELS_DIR / spec["model_file"]))
        it = spec.get("best_iteration")

        def predict(X):
            rng = (0, it + 1) if it is not None else (0, 0)
            return bst.predict(xgb.DMatrix(X), iteration_range=rng).astype(np.float32)
    else:
        bst = lgb.Booster(model_file=str(MODELS_DIR / spec["model_file"]))

        def predict(X):
            return bst.predict(X).astype(np.float32)
    stage2 = None
    if spec.get("stage2_file"):
        stage2 = xgb.Booster()
        stage2.load_model(str(MODELS_DIR / spec["stage2_file"]))
    return predict, stage2


def score_candidates(spec: dict, predict, stage2, ix: CountryIndex, cands: pd.DataFrame,
                     batch_entities: int = 150_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (all scored candidates, accepted matches), both with columns s1, cand, score."""
    import xgboost as xgb

    for b in ALL_BLOCKS:
        if f"rank_{b}" not in cands.columns:
            cands[f"rank_{b}"] = np.int16(-1)
    cands = cands.sort_values(["q", "c"], kind="stable").reset_index(drop=True)
    qv = cands["q"].to_numpy()
    uq = np.unique(qv)
    starts = np.searchsorted(qv, uq[::batch_entities])
    bounds = list(starts) + [len(cands)]
    all_c, all_m = [], []
    for i in range(len(bounds) - 1):
        sub = cands.iloc[bounds[i]:bounds[i + 1]]
        feats = build_features(ix, sub, ALL_BLOCKS)
        p = predict(feats[spec["features"]])
        if spec.get("calibration") == "isotonic":
            p = np.interp(p, spec["isotonic"]["x"], spec["isotonic"]["y"]).astype(np.float32)
        ent = pd.Index(pd.unique(feats["s1"].to_numpy()))
        ent_idx = ent.get_indexer(feats["s1"].to_numpy())
        pol = spec["policy"]
        if pol["type"] == "stage2":
            p = stage2.predict(xgb.DMatrix(context_matrix(ent_idx, p, len(ent), feats))).astype(np.float32)
            keep = p >= pol["t"]
        else:
            keep = policy_mask(pol, p, ent_idx, len(ent), feats["b_is_s3"].to_numpy())
        scored = pd.DataFrame({"s1": feats["s1"].to_numpy(), "cand": feats["cand"].to_numpy(), "score": p})
        all_c.append(scored)
        all_m.append(scored[keep])
        del feats
    return pd.concat(all_c, ignore_index=True), pd.concat(all_m, ignore_index=True)


def run_ml_inference(spec_file: str = "matcher_spec.json", split: str = "test", out_dir=OUTPUT_DIR,
                     validate: bool = True) -> dict:
    t0 = time.time()
    spec = json.loads((MODELS_DIR / spec_file).read_text(encoding="utf-8"))
    nz = learn_normalizer(folds()["s1"], truth_pairs(), tag=spec["normalizer_tag"])
    ids = normalized(split, nz, columns=["code", "source"])
    s1_codes = ids.loc[ids["source"] == 1, "code"].to_numpy()          # file order of Source 1
    del ids
    weak = json.loads(cache_path("records", f"{split}_weak_{nz['tag']}.json").read_text())
    predict, stage2 = load_matcher(spec)
    blocks = {b: k for b, k in spec["blocks"].items() if k > 0}
    cands_all, matches_all, timing = [], [], {}
    for country in countries(split, nz):          # open set: whatever labels the split contains
        tc = time.time()
        ix = CountryIndex(normalized_country(split, nz, country, INDEX_COLUMNS), set(weak.get(country, [])),
                          nz["maps"].get(country, {}).get("addr", {}))
        t_index = time.time() - tc
        q_rows = np.flatnonzero(ix.source == 1).astype(np.int32)
        cands = generate_candidates(ix, q_rows, blocks)
        t_block = time.time() - tc - t_index
        c, m = score_candidates(spec, predict, stage2, ix, cands)
        cands_all.append(c)
        matches_all.append(m)
        timing[country] = {"s1": int(len(q_rows)), "pool": int((ix.source != 1).sum()), "index_s": round(t_index),
                           "blocking_s": round(t_block), "score_s": round(time.time() - tc - t_index - t_block),
                           "candidates": int(len(c)), "matches": int(len(m))}
        print(f"[infer] {country}: {timing[country]}", flush=True)
        del ix, cands, c, m
    cands_all = pd.concat(cands_all, ignore_index=True)
    matches_all = pd.concat(matches_all, ignore_index=True)
    stats = write_submission(out_dir, s1_codes, cands_all, matches_all)
    per_ent = matches_all.groupby("s1").size().reindex(s1_codes, fill_value=0)
    report = {
        "model": spec["name"], "policy": spec["policy"], "split": split, "s1_entities": int(len(s1_codes)),
        "countries": timing, "candidate_pairs": int(len(cands_all)),
        "avg_candidates": float(len(cands_all) / len(s1_codes)), "predicted_matches": int(len(matches_all)),
        "entities_zero": int((per_ent == 0).sum()), "entities_one": int((per_ent == 1).sum()),
        "entities_multi": int((per_ent >= 2).sum()), "max_matches": int(per_ent.max()),
        "matched_rate": float((per_ent > 0).mean()), "files": stats,
        "file_mb": {f: round((out_dir / f).stat().st_size / 1e6, 1) for f in ("matching_results.tsv", "candidate_pairs.tsv")},
        "total_s": time.time() - t0,
    }
    if validate and split == "test":
        code, out = run_validator(out_dir)
        report["validator_exit_code"] = code
        report["validator_output"] = out
    (out_dir / "run_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return report
