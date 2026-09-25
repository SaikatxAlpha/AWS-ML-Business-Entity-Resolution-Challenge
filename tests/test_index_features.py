import numpy as np
import pandas as pd

from business_entity_resolution.blocking.candidate_generator import generate_candidates
from business_entity_resolution.blocking.index import CountryIndex, IndexConfig
from business_entity_resolution.features.feature_builder import build_features, feature_columns
from business_entity_resolution.utils.ids import encode_ids

ROWS = [
    ("S1-1", "Apex Healthcare Private Limited", "A102, Anjali Chs, Andheri West, Mumbai, 400053", "apex healthcare private limited", "apex healthcare", "a102 anjali chs andheri west mumbai 400053", "102 400053"),
    ("S1-2", "Moore Bitwise Inc", "337 Oakland Avenue, Michigan City, IN", "moore bitwise inc", "moore bitwise", "337 oakland avenue michigan city in", "337"),
    ("S2-1", "APEX HEALTHCARE PVT LTD", "A102 ANJALI CHS, ANDHERI WEST, MUMBAI 400053", "apex healthcare private limited", "apex healthcare", "a102 anjali chs andheri west mumbai 400053", "102 400053"),
    ("S3-1", "Apex Helthcare", "Andheri West, Mumbai", "apex helthcare", "apex helthcare", "andheri west mumbai", ""),
    ("S2-2", "Moore Bitwise", "337 Oakland Ave, Michigan City, IN", "moore bitwise", "moore bitwise", "337 oakland avenue michigan city in", "337"),
    ("S3-2", "Zeta Foods LLC", "12 River Road, Salem, OR", "zeta foods llc", "zeta foods", "12 river road salem or", "12"),
    ("S3-3", "Moore Holdings", "900 Pine Street, Dallas, TX", "moore holdings", "moore holdings", "900 pine street dallas tx", "900"),
]


def make_index():
    df = pd.DataFrame(ROWS, columns=["id", "name_raw", "addr_raw", "name_norm", "name_core", "addr_norm", "nums"])
    df["code"] = encode_ids(df["id"])
    df["source"] = df["id"].str[1].astype(int)
    df["country"] = "X"
    cfg = IndexConfig(max_df={"fielded": 1.0, "addr": 1.0, "name_char": 1.0, "name_word": 1.0})
    return CountryIndex(df, weak_tokens={"private", "limited", "inc", "llc"}, addr_map={}, cfg=cfg, verbose=False)


def test_retrieval_and_union():
    ix = make_index()
    cands = generate_candidates(ix, np.array([0, 1]), {"fielded": 2, "addr": 2, "name_char": 2, "name_word": 2})
    top = cands.sort_values(["q", "rank_fielded"]).groupby("q").first()
    assert top.loc[0, "c"] == 2 and top.loc[1, "c"] == 4        # true counterparts ranked first
    assert (cands["n_blocks"] >= 1).all()
    assert not cands.duplicated(["q", "c"]).any()
    assert set(ix.source[cands["c"]]) <= {2, 3}                  # never S1 as candidate


def test_feature_values():
    ix = make_index()
    cands = pd.DataFrame({"q": np.array([0, 0, 1, 1], dtype=np.int32), "c": np.array([2, 5, 4, 6], dtype=np.int32),
                          "n_blocks": np.int8(1)})
    for b in ("fielded", "addr", "name_char", "name_word"):
        cands[f"rank_{b}"] = np.int16(0)
    f = build_features(ix, cands, ["fielded", "addr", "name_char", "name_word"]).set_index("cand")
    ids = dict(zip([r[0] for r in ROWS], encode_ids(pd.Series([r[0] for r in ROWS]))))
    good, bad = f.loc[ids["S2-1"]], f.loc[ids["S3-2"]]
    assert good["name_exact"] == 1 and good["pin_exact_match"] == 1 and good["house_number_match"] == 1
    assert good["fielded_cos"] > 0.99 and bad["fielded_cos"] < 0.3
    assert bad["num_conflict"] == 1 and bad["core_idf_cov_a"] == 0
    m2, m3 = f.loc[ids["S2-2"]], f.loc[ids["S3-3"]]
    assert m2["core_exact"] == 1 and m2["house_number_match"] == 1
    assert m3["core_idf_cov_a"] > 0 and m3["num_conflict"] == 1   # shares "moore" but conflicting number
    assert f[feature_columns(f)].notna().all().all()
