"""Stages 5–18 on the fold-0 split: multi-block blocking, pair features, learned matcher,
calibration, decision policy (abstention / cardinality / singleton protection), ablations.

Entity sets (disjoint, all stratified by country × cardinality):
  model  — training-fold S1 entities used to fit the pairwise model
  calib  — other training-fold S1 entities: early stopping, calibration, policy tuning
  val    — validation-fold S1 entities: reported metrics only (never used for fitting/tuning)
All queries are matched against the full train S2/S3 pool of their country.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ..blocking.candidate_generator import BLOCKING_VERSION, generate_candidates
from ..blocking.index import CountryIndex
from ..config import MODELS_DIR, cache_path
from ..decision.abstention import context_matrix, policy_mask, tune_policies
from ..decision.threshold_optimizer import evaluate_mask, prepare
from ..evaluation.experiments import log_experiment
from ..evaluation.metrics import evaluate
from ..evaluation.validation import cardinality_bucket, split, stratified_sample, truth_pairs
from ..features.feature_builder import FEATURE_VERSION, build_features, feature_columns
from ..features.source_features import context_features
from ..models.calibration import Calibrator, ece, reliability
from ..models.train import predict_lgb, predict_xgb, train_lgb, train_xgb
from ..preprocessing.records import INDEX_COLUMNS, PREPROCESSING_VERSION, countries, learn_normalizer, normalized_country

GEN_BLOCKS = {"fielded": 40, "addr": 25, "name_char": 25, "name_word": 15}   # generous: analysed, then restricted
BLOCKS = list(GEN_BLOCKS)
CONFIG_GRID = [
    {"fielded": 40, "addr": 25, "name_char": 25, "name_word": 15},
    {"fielded": 30, "addr": 20, "name_char": 20, "name_word": 10},
    {"fielded": 30, "addr": 15, "name_char": 15, "name_word": 10},
    {"fielded": 25, "addr": 15, "name_char": 15, "name_word": 5},
    {"fielded": 20, "addr": 10, "name_char": 10, "name_word": 5},
    {"fielded": 30, "addr": 15, "name_char": 15, "name_word": 0},
    {"fielded": 30, "addr": 15, "name_char": 0, "name_word": 0},
    {"fielded": 30, "addr": 0, "name_char": 0, "name_word": 0},
    {"fielded": 50, "addr": 0, "name_char": 0, "name_word": 0},
]
MAX_AVG_CANDIDATES = 50


def dev_sets(n_model=150_000, n_calib=50_000, n_val=50_000):
    tr, va = split(0)
    model = stratified_sample(tr, n_model, seed=21)
    rest = tr[~tr["s1"].isin(model["s1"])]
    calib = stratified_sample(rest, n_calib, seed=22)
    val = stratified_sample(va, n_val, seed=12)
    return {"model": model, "calib": calib, "val": val}


def label_pairs(df: pd.DataFrame, T: pd.DataFrame) -> np.ndarray:
    key = pd.MultiIndex.from_arrays([df["s1"].to_numpy(), df["cand"].to_numpy()])
    tkey = pd.MultiIndex.from_frame(T[["s1", "cand"]])
    return key.isin(tkey).astype(np.int8)


def weak_tokens_for(split_name: str, tag: str) -> dict:
    return json.loads(cache_path("records", f"{split_name}_weak_{tag}.json").read_text())


# ------------------------------------------------------------------------ stage 5–7: build
def build_dev_features(tag: str = "dev") -> dict:
    t_all = time.time()
    T = truth_pairs()
    sets = dev_sets()
    set_of = pd.concat([s.assign(set=k)[["s1", "set"]] for k, s in sets.items()]).set_index("s1")["set"]
    tr, _ = split(0)
    nz = learn_normalizer(tr["s1"], T, tag="f0")
    country_list = countries("train", nz)
    weak = weak_tokens_for("train", "f0")
    info = {"timings": {}, "pool": {}, "index_s": {}, "features_s": {}}
    for country in country_list:
        out_path = cache_path("features", f"{tag}_{FEATURE_VERSION}_{country}.parquet")
        if out_path.exists():
            continue
        t0 = time.time()
        rc = normalized_country("train", nz, country, INDEX_COLUMNS)
        ix = CountryIndex(rc, set(weak.get(country, [])), nz["maps"].get(country, {}).get("addr", {}))
        del rc
        info["index_s"][country] = time.time() - t0
        info["pool"][country] = int((ix.source != 1).sum())
        q_codes = set_of.index[set_of.index.isin(ix.code[ix.source == 1])].to_numpy()
        q_rows = ix.row_of_code.get_indexer(q_codes).astype(np.int32)
        tm = {}
        cands = generate_candidates(ix, q_rows, GEN_BLOCKS, timings=tm)
        info["timings"][country] = tm
        t1 = time.time()
        feats = build_features(ix, cands, BLOCKS, verbose=True)
        info["features_s"][country] = time.time() - t1
        feats["y"] = label_pairs(feats, T)
        feats["set"] = set_of.reindex(feats["s1"]).to_numpy()
        feats["country"] = country
        feats.to_parquet(out_path, index=False)
        print(f"[build] {country}: {len(q_rows):,} queries, {len(feats):,} pairs, pos={int(feats['y'].sum()):,}; "
              f"blocks {({k: round(v) for k, v in tm.items()})}s; total {time.time() - t0:.0f}s", flush=True)
        del ix, cands, feats
    info["total_s"] = time.time() - t_all
    path = cache_path("features", f"{tag}_{FEATURE_VERSION}_info.json")
    if info["timings"]:
        path.write_text(json.dumps(info, indent=1))
    return json.loads(path.read_text())


def feature_files(tag: str = "dev") -> list:
    return sorted(cache_path("features", "x").parent.glob(f"{tag}_{FEATURE_VERSION}_*.parquet"))


def load_features(tag: str = "dev", columns=None, cfg: dict | None = None) -> pd.DataFrame:
    """Load per-country feature files; if ``cfg`` is given, restrict each file before concatenating."""
    parts = []
    for p in feature_files(tag):
        df = pd.read_parquet(p, columns=columns)
        if cfg is not None:
            df = restrict(df, cfg)
        for c in ("set", "country"):
            if c in df.columns:
                df[c] = df[c].astype("category")
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------------ stage 5–6: blocking analysis
def restrict(df: pd.DataFrame, cfg: dict, recompute_context: bool = True) -> pd.DataFrame:
    """Keep pairs retrieved under per-block K (``cfg``) and recompute provenance/context features."""
    keep = np.zeros(len(df), dtype=bool)
    nb = np.zeros(len(df), dtype=np.float32)
    ranks = {}
    for b in BLOCKS:
        r = df[f"rank_{b}"].to_numpy()
        m = r < cfg.get(b, 0)
        keep |= m
        nb += m
        ranks[b] = np.where(m, r, 99).astype(np.float32)
    out = df.loc[keep].copy()
    for b in BLOCKS:
        out[f"rank_{b}"] = ranks[b][keep]
    out["n_blocks"] = nb[keep]
    if recompute_context:
        ctx = context_features(out["s1"].to_numpy(), out)
        for k, v in ctx.items():
            out[k] = v
    return out.reset_index(drop=True)


def blocking_analysis(tag: str = "dev") -> dict:
    T = truth_pairs()
    info = json.loads(cache_path("features", f"{tag}_{FEATURE_VERSION}_info.json").read_text())
    cols = ["s1", "cand", "y", "set", "country"] + [f"rank_{b}" for b in BLOCKS]
    d = load_features(tag, cols)
    d = d[d["set"] == "val"]
    val = dev_sets()["val"]
    n_true = int(T["s1"].isin(val["s1"]).sum())
    n_ent = len(val)
    pool = sum(info["pool"].values())
    res = {"blocks": {}, "configs": []}
    for b in BLOCKS:
        r = d[f"rank_{b}"].to_numpy()
        for k in sorted({5, 10, 15, 20, 25, 30, 40, GEN_BLOCKS[b]}):
            if k > GEN_BLOCKS[b]:
                continue
            m = (r >= 0) & (r < k) & (r != 99)
            res["blocks"].setdefault(b, {})[k] = {"recall": float(d["y"].to_numpy()[m].sum() / n_true),
                                                  "avg_candidates": float(m.sum() / n_ent)}
        t_sec = sum(tm.get(b, 0) for tm in info["timings"].values())
        kmax = GEN_BLOCKS[b]
        rr = res["blocks"][b][kmax]
        log_experiment(f"BLK_{b}_k{kmax}", stage="5-blocking", metrics={},
                       cand_metrics={"candidate_recall": rr["recall"], "avg_candidates": rr["avg_candidates"]},
                       preprocessing_version=PREPROCESSING_VERSION, blocking_version=f"{BLOCKING_VERSION}:{b}",
                       model="retrieval only", hyperparameters={"k": kmax}, runtime_s=t_sec,
                       notes=f"single block on {n_ent:,} val entities; reduction ratio "
                             f"{1 - rr['avg_candidates'] / (pool / 2):.6f} vs mean per-country pool; "
                             f"runtime = retrieval for all dev queries ({sum(len(s) for s in dev_sets().values()):,})",
                       extra={"recall_at_k": res["blocks"][b]})
    for cfg in CONFIG_GRID:
        m = np.zeros(len(d), dtype=bool)
        for b in BLOCKS:
            r = d[f"rank_{b}"].to_numpy()
            m |= (r >= 0) & (r < cfg.get(b, 0))
        res["configs"].append({"cfg": cfg, "recall": float(d["y"].to_numpy()[m].sum() / n_true),
                               "avg_candidates": float(m.sum() / n_ent)})
    ok = [c for c in res["configs"] if c["avg_candidates"] <= MAX_AVG_CANDIDATES]
    best_recall = max(c["recall"] for c in ok)
    chosen = min((c for c in ok if c["recall"] >= best_recall - 0.001), key=lambda c: c["avg_candidates"])
    res["chosen"] = chosen
    for c in res["configs"]:
        name = "UNION_" + "_".join(f"{b[0]}{c['cfg'].get(b, 0)}" for b in BLOCKS)
        log_experiment(name, stage="5-blocking", metrics={},
                       cand_metrics={"candidate_recall": c["recall"], "avg_candidates": c["avg_candidates"]},
                       preprocessing_version=PREPROCESSING_VERSION, blocking_version=BLOCKING_VERSION,
                       model="retrieval only", hyperparameters=c["cfg"],
                       notes=("CHOSEN: " if c is chosen else "") + f"union of blocks on {n_ent:,} val entities; "
                             f"rule: best recall with avg<= {MAX_AVG_CANDIDATES}, fewest candidates within 0.1pp")
    (MODELS_DIR / "blocking_config.json").write_text(json.dumps(chosen, indent=1))
    print(f"[blocking] chosen {chosen}", flush=True)
    return res


# ------------------------------------------------------------------------ stage 8–14: models + decisions
def _eval_val(df_val, keep, T, val_ids, groups):
    return evaluate(df_val.loc[keep, ["s1", "cand"]], T, val_ids, groups)


def train_and_select(tag: str = "dev", ablation: bool = True) -> dict:
    T = truth_pairs()
    sets = dev_sets()
    cfg = json.loads((MODELS_DIR / "blocking_config.json").read_text())["cfg"]
    d = load_features(tag, cfg=cfg)
    fcols = feature_columns(d)
    tr, ca, va = (d[d["set"] == k].reset_index(drop=True) for k in ("model", "calib", "val"))
    del d
    print(f"[train] pairs model={len(tr):,} calib={len(ca):,} val={len(va):,}; features={len(fcols)}; "
          f"pos rate model={tr['y'].mean():.4f}", flush=True)
    val_ids, cal_ids = sets["val"]["s1"], sets["calib"]["s1"]
    groups = (sets["val"]["country"] + "|" + cardinality_bucket(sets["val"]["card"])).set_axis(sets["val"]["s1"])
    prep_ca, prep_va = prepare(ca, T, cal_ids), prepare(va, T, val_ids)
    cm_val = {"candidate_recall": float(va["y"].sum() / T["s1"].isin(val_ids).sum()), "avg_candidates": len(va) / len(val_ids)}
    common = dict(preprocessing_version=PREPROCESSING_VERSION, blocking_version=f"{BLOCKING_VERSION}:{cfg}",
                  feature_version=FEATURE_VERSION, cand_metrics=cm_val)
    results = {"n_pairs": {"model": len(tr), "calib": len(ca), "val": len(va)}, "features": fcols,
               "class_balance": {"model_pos_rate": float(tr["y"].mean()), "neg_per_pos": float((1 - tr["y"].mean()) / tr["y"].mean())}}

    # --- XGBoost vs LightGBM (same features, same split, same evaluator, global threshold tuned on calib)
    models = {}
    for name, fit, pred in (("xgb", train_xgb, predict_xgb), ("lgbm", train_lgb, predict_lgb)):
        bst, meta = fit(tr[fcols], tr["y"].to_numpy(), ca[fcols], ca["y"].to_numpy())
        t = time.time()
        p_ca, p_va = pred(bst, ca[fcols]), pred(bst, va[fcols])
        meta["predict_s_val"] = time.time() - t
        pols = tune_policies(prep_ca, p_ca, ca["b_is_s3"].to_numpy())
        g = pols["global"][0]
        m = _eval_val(va, p_va >= g["t"], T, val_ids, groups)
        models[name] = {"bst": bst, "meta": meta, "p_ca": p_ca, "p_va": p_va, "pols": pols, "val_global": m,
                        "calib_global_f05": pols["global"][1]["f05"]}
        log_experiment(f"M_{name}_global_{tag}", stage="8/12-model", metrics=m, model=name,
                       hyperparameters={k: v for k, v in meta["params"].items() if k not in ("nthread", "num_threads")} |
                       {"best_iteration": meta["best_iteration"]}, threshold=g["t"], runtime_s=meta["train_s"],
                       notes=f"global threshold tuned on calib (calib F0.5={pols['global'][1]['f05']:.4f}); "
                             f"train pairs {len(tr):,}", **common)
        print(f"[{name}] best_iter={meta['best_iteration']} train {meta['train_s']:.0f}s | calib F0.5={pols['global'][1]['f05']:.4f} "
              f"| val F0.5={m['f05']:.4f} P={m['micro_precision']:.4f} R={m['micro_recall']:.4f} "
              f"single={m['singleton_accuracy']:.4f} (t={g['t']})", flush=True)
    best_name = max(models, key=lambda k: models[k]["calib_global_f05"])
    B = models[best_name]
    results["model_comparison"] = {k: {"calib_f05": v["calib_global_f05"], "val_f05": v["val_global"]["f05"],
                                       "best_iteration": v["meta"]["best_iteration"], "train_s": v["meta"]["train_s"]}
                                   for k, v in models.items()}
    results["selected_model"] = best_name
    p_ca, p_va = B["p_ca"], B["p_va"]

    # --- calibration (cross-fitted on calib halves; kept only if it improves decisions)
    rng = np.random.default_rng(0)
    half = pd.Series(rng.random(len(cal_ids)) < 0.5, index=cal_ids.to_numpy())
    in_a = half.reindex(ca["s1"]).to_numpy()
    yc = ca["y"].to_numpy()
    calib_res = {"raw": {"ece_calib_half_b": ece(p_ca[~in_a], yc[~in_a])}}
    for kind in ("platt", "isotonic"):
        cal = Calibrator(kind).fit(p_ca[in_a], yc[in_a])
        calib_res[kind] = {"ece_calib_half_b": ece(cal.transform(p_ca[~in_a]), yc[~in_a])}
    results["calibration"] = calib_res
    results["reliability_raw_val"] = reliability(p_va, va["y"].to_numpy())
    iso = Calibrator("isotonic").fit(p_ca, yc)
    q_ca, q_va = iso.transform(p_ca), iso.transform(p_va)

    # --- decision policies (tuned on calib, reported on val) for raw and isotonic probabilities
    policy_rows = {}
    for pname, (pc, pv) in {"raw": (p_ca, p_va), "isotonic": (q_ca, q_va)}.items():
        pols = B["pols"] if pname == "raw" else tune_policies(prep_ca, pc, ca["b_is_s3"].to_numpy())
        for kind, (pol, calm) in pols.items():
            keep = policy_mask(pol, pv, prep_va.ent_idx, len(prep_va.n_true), va["b_is_s3"].to_numpy())
            m = _eval_val(va, keep, T, val_ids, groups)
            policy_rows[f"{pname}/{kind}"] = {"policy": pol, "calib_f05": calm["f05"], "val": m}
            log_experiment(f"D_{best_name}_{pname}_{kind}_{tag}", stage="11-14-decision", metrics=m, model=best_name,
                           hyperparameters={"calibration": pname}, threshold=pol,
                           notes=f"policy '{kind}' tuned on calib (calib F0.5={calm['f05']:.4f})", **common)
            print(f"[policy] {pname:8s} {kind:9s} calib={calm['f05']:.4f} val={m['f05']:.4f} "
                  f"single={m['singleton_accuracy']:.4f} {pol}", flush=True)

    # --- stage-2 entity-context model (cross-fitted on calib), singleton protection by learning
    Xc = context_matrix(prep_ca.ent_idx, p_ca, len(prep_ca.n_true), ca)
    Xv = context_matrix(prep_va.ent_idx, p_va, len(prep_va.n_true), va)
    oof = np.zeros(len(ca), dtype=np.float32)
    iters = []
    for fold in (True, False):
        tr_m, te_m = in_a == fold, in_a != fold
        b2, m2meta = train_xgb(Xc[tr_m], yc[tr_m], Xc[te_m], yc[te_m], params={"max_depth": 6}, rounds=600, early=30)
        oof[te_m] = predict_xgb(b2, Xc[te_m])
        iters.append(m2meta["best_iteration"] + 1)
    grid = np.round(np.arange(0.05, 0.991, 0.01), 3)
    t2 = max(grid, key=lambda t: evaluate_mask(prep_ca, oof >= t)["f05"])
    s2_cal = evaluate_mask(prep_ca, oof >= t2)
    b2_full, meta2 = train_xgb(Xc, yc, Xc, yc, params={"max_depth": 6}, rounds=int(np.mean(iters)), early=10_000)
    p2_va = predict_xgb(b2_full, Xv)
    m2 = _eval_val(va, p2_va >= t2, T, val_ids, groups)
    policy_rows["stage2/global"] = {"policy": {"type": "stage2", "t": float(t2)}, "calib_f05": s2_cal["f05"], "val": m2}
    log_experiment(f"D_{best_name}_stage2_{tag}", stage="11-14-decision", metrics=m2, model=f"{best_name}+stage2-xgb",
                   threshold=float(t2), notes=f"entity-context second model, 2-fold cross-fitted on calib "
                                             f"(calib OOF F0.5={s2_cal['f05']:.4f})", **common)
    print(f"[policy] stage2            calib={s2_cal['f05']:.4f} val={m2['f05']:.4f} single={m2['singleton_accuracy']:.4f} t={t2}", flush=True)
    results["policies"] = {k: {"policy": v["policy"], "calib_f05": v["calib_f05"], "val_f05": v["val"]["f05"],
                               "val": v["val"]} for k, v in policy_rows.items()}

    # --- final selection on calib F0.5 only
    sel = max(policy_rows, key=lambda k: policy_rows[k]["calib_f05"])
    results["selected_policy"] = sel
    spec = {
        "name": f"matcher_{best_name}", "type": "ml", "normalizer_tag": "f0", "blocks": cfg,
        "feature_version": FEATURE_VERSION, "preprocessing_version": PREPROCESSING_VERSION,
        "features": fcols, "model": best_name, "model_file": f"matcher_{best_name}.model",
        "calibration": "isotonic" if sel.startswith("isotonic") else "none",
        "policy": policy_rows[sel]["policy"],
        "stage2_file": "matcher_stage2.json" if sel.startswith("stage2") else None,
        "val_metrics": policy_rows[sel]["val"], "calib_f05": policy_rows[sel]["calib_f05"],
        "selection": "model and policy selected on calib F0.5; val only reported",
    }
    if best_name == "xgb":
        B["bst"].save_model(str(MODELS_DIR / spec["model_file"]))
        spec["best_iteration"] = B["meta"]["best_iteration"]
    else:
        B["bst"].save_model(str(MODELS_DIR / spec["model_file"]), num_iteration=B["meta"]["best_iteration"])
    if spec["calibration"] == "isotonic":
        spec["isotonic"] = {"x": iso.model.X_thresholds_.tolist(), "y": iso.model.y_thresholds_.tolist()}
    if spec["stage2_file"]:
        b2_full.save_model(str(MODELS_DIR / spec["stage2_file"]))
    (MODELS_DIR / "matcher_spec.json").write_text(json.dumps(spec, indent=1, default=str))
    print(f"[select] model={best_name} policy={sel} calib={policy_rows[sel]['calib_f05']:.4f} "
          f"val={policy_rows[sel]['val']['f05']:.4f}", flush=True)

    imp = B["bst"].get_score(importance_type="gain") if best_name == "xgb" else dict(zip(fcols, B["bst"].feature_importance("gain")))
    results["feature_importance_gain"] = dict(sorted(((k, float(v)) for k, v in imp.items()), key=lambda x: -x[1]))
    if ablation:
        results["ablation"] = run_ablation(tr, ca, va, fcols, prep_ca, T, val_ids, groups, common, tag)
    (cache_path("features", "x").parent / f"{tag}_train_results.json").write_text(json.dumps(results, indent=1, default=str))
    return results


# ------------------------------------------------------------------------ stage 18: ablation
GROUPS = {
    "name_string": ["name_exact", "core_exact", "name_ratio", "core_ratio", "name_jw", "core_dl", "name_tsort",
                    "name_tset", "core_partial", "name_common", "name_jacc", "name_dice", "name_overlap",
                    "core_common", "core_jacc", "name_len_ratio", "name_ntok_diff", "core_ntok_a", "core_ntok_b",
                    "first_core_eq", "acronym_match", "b_name_nonascii", "b_name_domain", "b_name_alias"],
    "address": ["addr_exact", "addr_ratio", "addr_tset", "addr_tsort", "addr_common", "addr_jacc", "comp_common",
                "comp_cov_a", "comp_jacc", "last_comp_eq", "addr_ntok_a", "addr_ntok_b", "b_addr_empty", "b_addr_nonascii"],
    "numeric": ["num_common", "num_jacc", "num_cov_a", "num_conflict", "num_cnt_a", "num_cnt_b", "num_cnt_diff",
                "house_number_match", "first_num_eq", "pin_both", "pin_exact_match", "pin_conflict"],
    "rare_token_idf": ["core_idf_cov_a", "core_idf_cov_b", "core_idf_shared", "core_max_shared_idf", "name_word_cos",
                       "name_char_cos", "addr_idf_cov_a", "addr_idf_cov_b", "addr_cos", "fielded_cos", "num_idf_shared",
                       "name_generic_a", "name_generic_b", "addr_generic_a", "addr_generic_b"],
    "block_agreement": ["n_blocks", "b_is_s3"] + [f"rank_{b}" for b in BLOCKS],
    "context_graph": None,   # everything remaining: candidate-list context + top-candidate consistency
}


def run_ablation(tr, ca, va, fcols, prep_ca, T, val_ids, groups, common, tag, n_model_entities: int = 50_000) -> list:
    rng = np.random.default_rng(1)
    ents = tr["s1"].unique()
    sub_ids = rng.choice(ents, size=min(n_model_entities, len(ents)), replace=False)
    trs = tr[tr["s1"].isin(sub_ids)]
    used, rows = [], []
    assigned = {c for g in GROUPS.values() if g for c in g}
    for gname, cols in GROUPS.items():
        cols = [c for c in fcols if c not in assigned] if cols is None else [c for c in cols if c in fcols]
        used = used + cols
        bst, meta = train_xgb(trs[used], trs["y"].to_numpy(), ca[used], ca["y"].to_numpy(), rounds=800, early=30)
        p_ca, p_va = predict_xgb(bst, ca[used]), predict_xgb(bst, va[used])
        grid = np.round(np.arange(0.05, 0.991, 0.01), 3)
        t = max(grid, key=lambda x: evaluate_mask(prep_ca, p_ca >= x)["f05"])
        m = evaluate(va.loc[p_va >= t, ["s1", "cand"]], T, val_ids, groups)
        rows.append({"added_group": gname, "n_features": len(used), "val_f05": m["f05"],
                     "val_precision": m["micro_precision"], "val_recall": m["micro_recall"],
                     "singleton_accuracy": m["singleton_accuracy"], "threshold": float(t)})
        log_experiment(f"ABL_{len(rows):02d}_{gname}_{tag}", stage="18-ablation", metrics=m, model="xgb",
                       hyperparameters={"n_features": len(used), "model_entities": len(sub_ids)}, threshold=float(t),
                       runtime_s=meta["train_s"], notes=f"cumulative feature groups up to '{gname}'; global threshold on calib",
                       **common)
        print(f"[ablation] +{gname:16s} n={len(used):3d} val F0.5={m['f05']:.4f}", flush=True)
    return rows


# ------------------------------------------------------------------------ final check: full validation fold
def evaluate_full_validation(spec_file: str = "matcher_spec.json") -> dict:
    """Score ALL fold-0 validation entities with the exact inference code path and the saved spec."""
    from ..pipeline.ml_inference import load_matcher, score_candidates

    t0 = time.time()
    T = truth_pairs()
    _, va = split(0)
    spec = json.loads((MODELS_DIR / spec_file).read_text(encoding="utf-8"))
    nz = learn_normalizer(split(0)[0]["s1"], T, tag=spec["normalizer_tag"])
    weak = weak_tokens_for("train", nz["tag"])
    predict, stage2 = load_matcher(spec)
    blocks = {b: k for b, k in spec["blocks"].items() if k > 0}
    cands_all, matches_all = [], []
    for country in countries("train", nz):
        ix = CountryIndex(normalized_country("train", nz, country, INDEX_COLUMNS), set(weak.get(country, [])),
                          nz["maps"].get(country, {}).get("addr", {}))
        q_codes = va.loc[va["country"] == country, "s1"].to_numpy()
        q_rows = ix.row_of_code.get_indexer(q_codes).astype(np.int32)
        cands = generate_candidates(ix, q_rows, blocks)
        c, m = score_candidates(spec, predict, stage2, ix, cands)
        cands_all.append(c[["s1", "cand"]])
        matches_all.append(m[["s1", "cand"]])
        print(f"[fullval] {country}: {len(q_rows):,} entities, {len(c):,} candidates, {len(m):,} matches", flush=True)
        del ix, cands, c, m
    cands_all = pd.concat(cands_all, ignore_index=True)
    matches_all = pd.concat(matches_all, ignore_index=True)
    from ..evaluation.metrics import candidate_metrics
    groups = (va["country"] + "|" + cardinality_bucket(va["card"])).set_axis(va["s1"])
    m = evaluate(matches_all, T, va["s1"], groups)
    cm = candidate_metrics(cands_all, T, va["s1"])
    runtime = time.time() - t0
    log_experiment(f"FINAL_fullval_{spec['name']}", stage="20-final", metrics=m, cand_metrics=cm,
                   preprocessing_version=spec["preprocessing_version"], blocking_version=f"{BLOCKING_VERSION}:{spec['blocks']}",
                   feature_version=spec["feature_version"], model=spec["model"], threshold=spec["policy"],
                   runtime_s=runtime, notes=f"ALL {len(va):,} fold-0 validation entities, exact inference code path")
    out = {"metrics": m, "candidates": cm, "runtime_s": runtime}
    (cache_path("features", "x").parent / "fullval_results.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"[fullval] F0.5={m['f05']:.4f} P={m['micro_precision']:.4f} R={m['micro_recall']:.4f} "
          f"single={m['singleton_accuracy']:.4f} cand_recall={cm['candidate_recall']:.4f} avg_cands={cm['avg_candidates']:.1f}", flush=True)
    return out
