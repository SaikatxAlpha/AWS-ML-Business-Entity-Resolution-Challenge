"""Stage 4 — transparent baseline experiments (logged to experiments/results.csv).

Protocol (leakage-safe):
* normaliser (token maps, weak tokens) learned from training-fold positives only;
* retrieval statistics (IDF) from the split's records, label-free;
* baseline parameters (w, t) tuned on a stratified sample of *training-fold* S1 entities;
* metrics reported on *validation-fold* S1 entities, retrieved against the full train S2/S3 pool.

Variants:
  E000  empty prediction for every entity (floor = singleton rate)
  B0    raw text: lowercase + split on non-alphanumerics, no transliteration/maps/weak tokens
  B1    full normalisation (p1): transliteration, OCR repair, learned maps, core names, numbers
For each: fuzzy (w, t) rule and a TF-IDF-score-only rule.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ..blocking.tfidf_blocking import TfidfBlockingConfig, tfidf_candidates
from ..config import EXPERIMENTS_DIR, cache_path
from ..decision.threshold_optimizer import best, evaluate_mask, grid_search, prepare
from ..evaluation.experiments import log_experiment
from ..evaluation.metrics import candidate_metrics, evaluate
from ..evaluation.validation import cardinality_bucket, split, stratified_sample, truth_pairs
from ..models.baseline import combined, fuzzy_scores, tune
from ..preprocessing.records import PREPROCESSING_VERSION, learn_normalizer, normalized

K_LIST = (1, 5, 10, 20, 50)


def _raw_tokens(s: pd.Series) -> pd.Series:
    return s.str.lower().str.replace(r"[^0-9a-z]+", " ", regex=True).str.strip()


VARIANTS = {
    "B0_raw": dict(
        preprocessing="raw-lower",
        fields={"n": "name_rawtok", "a": "addr_rawtok"},
        name_col="name_rawtok", addr_col="addr_rawtok",
    ),
    "B1_norm": dict(
        preprocessing=PREPROCESSING_VERSION,
        fields={"n": "name_core", "a": "addr_norm", "d": "nums"},
        name_col="name_core", addr_col="addr_norm",
    ),
}


def run_baseline(eval_n: int | None = 50_000, tune_n: int = 50_000, top_k: int = 50,
                 variants=("B0_raw", "B1_norm"), tag: str = "dev") -> dict:
    t_all = time.time()
    T = truth_pairs()
    tr, va = split(0)
    tune_set = stratified_sample(tr, tune_n, seed=11)
    eval_set = va if eval_n is None else stratified_sample(va, eval_n, seed=12)
    groups = (eval_set["country"] + "|" + cardinality_bucket(eval_set["card"])).set_axis(eval_set["s1"])
    n_eval = len(eval_set)
    split_note = f"fold0 val entities={n_eval:,}; tune=train-fold sample {len(tune_set):,}"

    empty = pd.DataFrame({"s1": pd.Series(dtype=np.int64), "cand": pd.Series(dtype=np.int64)})
    m0 = evaluate(empty, T, eval_set["s1"], groups)
    log_experiment(f"E000_empty_{tag}", metrics=m0, model="empty-prediction",
                   notes=f"floor: predict no match for every entity; {split_note}")

    nz = learn_normalizer(tr["s1"], T, tag="f0")
    rec = normalized("train", nz)
    rec["name_rawtok"] = _raw_tokens(rec["name_raw"])
    rec["addr_rawtok"] = _raw_tokens(rec["addr_raw"])
    queries = np.concatenate([tune_set["s1"].to_numpy(), eval_set["s1"].to_numpy()])

    results = {}
    for name in variants:
        v = VARIANTS[name]
        t0 = time.time()
        cfg = TfidfBlockingConfig(fields=v["fields"], top_k=top_k)
        cpath = cache_path("candidates", f"baseline_{name}_{tag}_k{top_k}.parquet")
        if cpath.exists():
            cands = pd.read_parquet(cpath)
        else:
            cands = tfidf_candidates(rec, queries, cfg)
            cands.to_parquet(cpath, index=False)
        t_block = time.time() - t0
        ev_c = cands[cands["s1"].isin(eval_set["s1"])]
        cand_at_k = {k: candidate_metrics(ev_c[ev_c["rank"] < k], T, eval_set["s1"]) for k in K_LIST}
        cm = cand_at_k[top_k]
        print(f"[{name}] blocking {t_block:.0f}s; recall@K " +
              ", ".join(f"{k}:{cand_at_k[k]['candidate_recall']:.4f}" for k in K_LIST), flush=True)

        t1 = time.time()
        scored = fuzzy_scores(cands, rec, v["name_col"], v["addr_col"])
        tu = scored[scored["s1"].isin(tune_set["s1"])].reset_index(drop=True)
        ev = scored[scored["s1"].isin(eval_set["s1"])].reset_index(drop=True)
        prep_tu, prep_ev = prepare(tu, T, tune_set["s1"]), prepare(ev, T, eval_set["s1"])

        # rule 1: fuzzy combination
        params, curves = tune(tu, prep_tu)
        keep = combined(ev, params["w"]) >= params["t"]
        fast = evaluate_mask(prep_ev, keep)
        full = evaluate(ev.loc[keep, ["s1", "cand"]], T, eval_set["s1"], groups)
        assert abs(fast["f05"] - full["f05"]) < 1e-9, "fast/reference evaluator mismatch"
        runtime = time.time() - t0
        hp = {"top_k": top_k, "max_df": cfg.max_df, "fields": v["fields"], "w": params["w"]}
        log_experiment(f"{name}_fuzzy_{tag}", metrics=full, cand_metrics=cm,
                       preprocessing_version=v["preprocessing"], blocking_version="tfidf_fielded_v0",
                       feature_version="fuzzy_tokenset_v0", model="rule: w*name+(1-w)*addr >= t",
                       hyperparameters=hp, threshold=params["t"], runtime_s=runtime,
                       notes=f"(w,t) tuned on train-fold sample; {split_note}",
                       extra={"candidate_recall_at_k": cand_at_k, "tune_best": best(curves).to_dict()})
        curves.to_csv(EXPERIMENTS_DIR / "runs" / f"{name}_fuzzy_{tag}_tuning_curve.csv", index=False)

        # rule 2: retrieval score only
        c2 = grid_search(prep_tu, tu["score"].to_numpy(), np.round(np.arange(0.05, 1.0, 0.01), 2))
        t2 = float(best(c2)["threshold"])
        keep2 = ev["score"].to_numpy() >= t2
        full2 = evaluate(ev.loc[keep2, ["s1", "cand"]], T, eval_set["s1"], groups)
        log_experiment(f"{name}_tfidfonly_{tag}", metrics=full2, cand_metrics=cm,
                       preprocessing_version=v["preprocessing"], blocking_version="tfidf_fielded_v0",
                       feature_version="none", model="rule: tfidf cosine >= t",
                       hyperparameters={"top_k": top_k, "max_df": cfg.max_df, "fields": v["fields"]},
                       threshold=t2, runtime_s=time.time() - t0, notes=f"threshold tuned on train-fold sample; {split_note}")
        results[name] = {"fuzzy": full, "fuzzy_params": params, "tfidf_only": full2, "tfidf_threshold": t2,
                         "candidates": cand_at_k, "scoring_s": time.time() - t1, "blocking_s": t_block}
        print(f"[{name}] fuzzy F0.5={full['f05']:.4f} (w={params['w']}, t={params['t']}) | "
              f"tfidf-only F0.5={full2['f05']:.4f} (t={t2}) | {time.time() - t0:.0f}s", flush=True)

    results["empty"] = m0
    results["runtime_s"] = time.time() - t_all
    (EXPERIMENTS_DIR / "runs" / f"baseline_summary_{tag}.json").write_text(json.dumps(results, indent=1, default=str))
    return results
