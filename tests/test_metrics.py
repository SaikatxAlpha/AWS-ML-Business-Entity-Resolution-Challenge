import numpy as np
import pandas as pd
import pytest

from business_entity_resolution.evaluation.metrics import (
    candidate_metrics,
    entity_table,
    evaluate,
    f05_from_sets,
)
from business_entity_resolution.utils.ids import decode_ids, encode_ids


def pairs(d):
    return pd.DataFrame([(k, v) for k, vs in d.items() for v in vs], columns=["s1", "cand"])


def test_readme_example():
    pred = pairs({"S1-00001": ["S2-00047", "S2-00193", "S3-00812"]})
    truth = pairs({"S1-00001": ["S2-00047", "S3-00812"]})
    assert evaluate(pred, truth, ["S1-00001"])["f05"] == pytest.approx(0.7142857, abs=1e-6)


def test_singleton_and_missing_rules():
    truth = pairs({"a": ["x"]})  # "b" and "c" are singletons (no truth rows)
    pred = pairs({"b": ["y"]})   # false merge on singleton b; c predicted empty; a missing
    ent = entity_table(pred, truth, ["a", "b", "c"])
    assert ent.loc["a", "f05"] == 0.0      # true match exists, predicted nothing
    assert ent.loc["b", "f05"] == 0.0      # singleton with a prediction
    assert ent.loc["c", "f05"] == 1.0      # singleton predicted empty
    m = evaluate(pred, truth, ["a", "b", "c"])
    assert m["f05"] == pytest.approx(1 / 3)
    assert m["singleton_accuracy"] == pytest.approx(0.5)
    assert m["false_merge_rate"] == 1.0


def test_duplicates_and_out_of_scope_ignored():
    truth = pairs({"a": ["x", "y"]})
    pred = pairs({"a": ["x", "x"], "zzz": ["q"]})
    ent = entity_table(pred, truth, ["a"])
    assert ent.loc["a", "n_pred"] == 1
    assert ent.loc["a", "f05"] == pytest.approx(f05_from_sets({"x"}, {"x", "y"}))


def test_vectorised_matches_reference():
    rng = np.random.default_rng(0)
    ents = [f"e{i}" for i in range(300)]
    items = [f"r{i}" for i in range(40)]
    T = {e: set(rng.choice(items, rng.integers(0, 4), replace=False)) for e in ents}
    P = {e: set(rng.choice(items, rng.integers(0, 4), replace=False)) | (T[e] if rng.random() < .5 else set()) for e in ents}
    ref = np.mean([f05_from_sets(P[e], T[e]) for e in ents])
    assert evaluate(pairs(P), pairs(T), ents)["f05"] == pytest.approx(ref)


def test_candidate_metrics_oracle():
    truth = pairs({"a": ["x", "y"], "b": ["z"]})
    cand = pairs({"a": ["x", "w"], "b": ["z", "v"], "c": ["u"]})
    cm = candidate_metrics(cand, truth, ["a", "b", "c"])
    assert cm["candidate_recall"] == pytest.approx(2 / 3)
    assert cm["avg_candidates"] == pytest.approx(5 / 3)
    # oracle: a -> {x} of {x,y}; b -> {z}; c singleton -> empty
    expected = np.mean([f05_from_sets({"x"}, {"x", "y"}), 1.0, 1.0])
    assert cm["oracle_f05"] == pytest.approx(expected)


def test_id_codes_roundtrip():
    ids = pd.Series(["S1-965667", "S2-681193310", "S3-11291185"])
    codes = encode_ids(ids)
    assert decode_ids(codes).tolist() == ids.tolist()
    with pytest.raises(ValueError):
        encode_ids(pd.Series(["X1-1"]))


def test_fast_optimizer_matches_reference():
    from business_entity_resolution.decision.threshold_optimizer import evaluate_mask, prepare
    rng = np.random.default_rng(1)
    ents = np.arange(200, dtype=np.int64)
    items = np.arange(1000, 1060, dtype=np.int64)
    truth = pd.DataFrame([(e, i) for e in ents for i in rng.choice(items, rng.integers(0, 4), replace=False)],
                         columns=["s1", "cand"])
    cands = pd.concat([truth.sample(frac=0.8, random_state=0),
                       pd.DataFrame({"s1": rng.choice(ents, 500), "cand": rng.choice(items, 500)})]).drop_duplicates()
    scores = rng.random(len(cands))
    prep = prepare(cands, truth, ents)
    for t in (0.0, 0.3, 0.7, 1.1):
        keep = scores >= t
        fast = evaluate_mask(prep, keep)["f05"]
        ref = evaluate(cands[keep], truth, ents)["f05"]
        assert fast == pytest.approx(ref)
