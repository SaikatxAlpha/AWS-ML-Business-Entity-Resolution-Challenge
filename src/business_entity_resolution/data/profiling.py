"""Dataset forensics: every statistic in reports/dataset_profile.md is computed here.

Full-data statistics are vectorised with pandas (Arrow-backed strings). A few
character-level statistics (Unicode scripts, character inventory) use a fixed-seed
random sample of rows per file; those are labelled as sampled in the report.
"""
from __future__ import annotations

import os
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import psutil
from rapidfuzz import fuzz, process

from ..config import SEED, cache_path, source_path
from ..preprocessing.normalization import basic_normalize_series, unicode_script
from .loader import load_ground_truth, load_source
from .validation import check_frame, check_raw_file, check_ground_truth

QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
CHAR_SAMPLE = 200_000
PAIR_SAMPLE_S1 = 150_000

NAME_PATTERNS = {
    "non_ascii": r"[^\x00-\x7f]",
    "has_digit": r"\d",
    "ocr_digit_in_word": r"(?i)[a-z][018][a-z]|\b[018][a-z]{3,}",
    "ampersand": r"&",
    "acronym_only": r"^[A-Z]{2,5}$",
    "domain_name": r"(?i)\.(?:com|net|org|in|co|fr|biz|info)\b",
    "alias_marker": r"(?i)\b(?:f/k/a|fka|d/b/a|dba|a/k/a|aka|formerly|t/a)\b",
    "unusual_punct": r"[!?@*$%^_=+~|<>]",
    "multi_space": r"\s{2,}",
    "lead_trail_space": r"^\s|\s$",
}
ADDR_PATTERNS = {
    "empty": r"^\s*$",
    "non_ascii": r"[^\x00-\x7f]",
    "has_digit": r"\d",
    "token_5digit": r"(?<!\d)\d{5}(?!\d)",
    "token_6digit": r"(?<!\d)\d{6}(?!\d)",
    "spaced_pin_ddd_ddd": r"(?<!\d)\d{3} \d{3}(?!\d)",
    "zip_plus4": r"\b\d{5}-\d{4}\b",
    "starts_with_number": r"^\s*#?\s*\d",
    "hash_number": r"#\s?\d",
    "zero_padded_number": r"(?<![\d.])0\d+",
    "landmark_marker": r"(?i)\b(?:near|opp|opposite|behind|beside|next to|nr|adj|adjacent)\b",
    "unit_marker": r"(?i)\b(?:unit|apt|apartment|suite|ste|floor|flr|flat|fl)\b",
    "po_box": r"(?i)\bp\.?\s?o\.?\s?box\b",
    "null_literal": r"(?i)<null>|\bnull\b|\bn/a\b",
}


# --------------------------------------------------------------------------- helpers
def _quantiles(s: pd.Series) -> dict:
    q = s.quantile(QUANTILES)
    out = {f"p{int(k * 100)}": float(v) for k, v in q.items()}
    out["mean"] = float(s.mean())
    return out


def _pattern_rates(df: pd.DataFrame, col: str, patterns: dict) -> pd.DataFrame:
    flags = pd.DataFrame({k: df[col].str.contains(p, regex=True) for k, p in patterns.items()})
    flags["country"] = df["country"].values
    by = flags.groupby("country").mean()
    by.loc["ALL"] = flags.drop(columns="country").mean()
    return by.T


def _token_counts(norm: pd.Series) -> pd.Series:
    return norm.str.split().explode().value_counts()


# --------------------------------------------------------------------------- per file
def profile_file(split: str, k: int) -> dict:
    t0 = time.time()
    raw = check_raw_file(split, k)
    t_raw = time.time() - t0
    t0 = time.time()
    df = load_source(split, k)
    t_load = time.time() - t0
    out: dict = {
        "file": source_path(split, k).name,
        "size_mb": os.path.getsize(source_path(split, k)) / 1e6,
        "parquet_mb": os.path.getsize(cache_path("raw", f"{split}_source{k}.parquet")) / 1e6,
        "raw_scan_s": t_raw,
        "load_s": t_load,
        "mem_gb": df.memory_usage(deep=True).sum() / 1e9,
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "raw_lines": raw.n_lines,
        "field_counts": dict(raw.field_counts),
        "malformed_rows": raw.malformed,
        "rows_with_quote": raw.rows_with_quote,
        "rows_with_cr": raw.rows_with_cr,
        "bom": raw.has_bom,
        "frame_checks": check_frame(df, k),
        "country_counts": df["country"].value_counts().to_dict(),
    }
    name, addr = df["business_name"], df["business_address"]
    out["name_empty"] = int((name.str.strip() == "").sum())
    out["addr_empty"] = int((addr.str.strip() == "").sum())
    out["country_empty"] = int((df["country"].str.strip() == "").sum())

    out["name_chars"] = _quantiles(name.str.len())
    out["name_tokens"] = _quantiles(name.str.count(r"\S+"))
    nonempty_addr = addr[addr.str.strip() != ""]
    out["addr_chars"] = _quantiles(nonempty_addr.str.len())
    out["addr_tokens"] = _quantiles(nonempty_addr.str.count(r"\S+"))
    out["addr_components"] = _quantiles(nonempty_addr.str.count(",") + 1)
    out["median_len_by_country"] = (
        df.assign(nl=name.str.len(), al=addr.str.len())
        .groupby("country")[["nl", "al"]].median().to_dict()
    )

    upper = name.str.contains(r"[A-Za-z]") & (name == name.str.upper())
    lower = name.str.contains(r"[A-Za-z]") & (name == name.str.lower())
    out["name_case"] = {"all_upper": float(upper.mean()), "all_lower": float(lower.mean()),
                        "mixed": float(1 - upper.mean() - lower.mean())}
    out["name_patterns"] = _pattern_rates(df, "business_name", NAME_PATTERNS)
    out["addr_patterns"] = _pattern_rates(df, "business_address", ADDR_PATTERNS)

    # duplicates
    out["dup"] = {
        "distinct_names": int(name.nunique()),
        "rows_sharing_exact_name": int(name.duplicated(keep=False).sum()),
        "distinct_addresses": int(nonempty_addr.nunique()),
        "rows_sharing_exact_address": int(nonempty_addr.duplicated(keep=False).sum()),
        "rows_sharing_name_and_address": int(df.duplicated(["business_name", "business_address"], keep=False).sum()),
        "rows_sharing_name_address_country": int(df.duplicated(["business_name", "business_address", "country"], keep=False).sum()),
    }
    out["top_names"] = name.value_counts().head(12).to_dict()
    out["top_addresses"] = nonempty_addr.value_counts().head(8).to_dict()

    # normalised tokens (full data)
    norm = basic_normalize_series(name)
    out["dup"]["rows_sharing_normalized_name"] = int(norm.duplicated(keep=False).sum())
    out["dup"]["distinct_normalized_names"] = int(norm.nunique())
    tok_by_country, last_by_country, vocab = {}, {}, {}
    for c, idx in df.groupby("country").groups.items():
        nc = norm.loc[idx]
        tc = _token_counts(nc)
        vocab[c] = int(len(tc))
        tok_by_country[c] = (tc.head(30) / len(nc)).round(4).to_dict()
        last_by_country[c] = (nc.str.rsplit(" ", n=1).str[-1].value_counts().head(20) / len(nc)).round(4).to_dict()
    out["name_vocab_size"] = vocab
    out["top_name_tokens_df"] = tok_by_country
    out["top_last_name_token"] = last_by_country

    # last address component (usually state/region) per country
    last_comp = addr.str.rsplit(",", n=1).str[-1].str.strip()
    out["top_last_addr_component"] = {
        c: (last_comp.loc[idx].value_counts().head(12) / len(idx)).round(4).to_dict()
        for c, idx in df.groupby("country").groups.items()
    }

    # sampled character-level statistics
    rng = np.random.default_rng(SEED + k + (0 if split == "train" else 10))
    samp = df.iloc[rng.choice(len(df), size=min(CHAR_SAMPLE, len(df)), replace=False)]
    for field in ("business_name", "business_address"):
        scripts, chars = Counter(), Counter()
        for text in samp[field]:
            if text.isascii():
                continue
            seen = set()
            for ch in text:
                if ord(ch) > 127:
                    chars[ch] += 1
                    seen.add(unicode_script(ch))
            scripts.update(seen)
        out[f"{field}_scripts_sampled"] = {s: v / len(samp) for s, v in scripts.most_common(15)}
        out[f"{field}_nonascii_chars_sampled"] = dict(chars.most_common(25))
    ascii_punct = Counter(ch for t in samp["business_name"] for ch in t if ch.isascii() and not ch.isalnum() and ch != " ")
    out["name_ascii_punct_sampled"] = dict(ascii_punct.most_common(20))
    out["examples"] = {
        c: samp[samp["country"] == c].head(5)[["entity_id", "business_name", "business_address"]].values.tolist()
        for c in sorted(samp["country"].unique())
    }
    return out


# --------------------------------------------------------------------------- ground truth
def profile_ground_truth() -> dict:
    s1 = load_source("train", 1)[["entity_id", "country", "business_name", "business_address"]]
    s2 = load_source("train", 2)[["entity_id", "country"]]
    s3 = load_source("train", 3)[["entity_id", "country"]]
    gt = load_ground_truth()
    s23_ids = set(s2["entity_id"]) | set(s3["entity_id"])
    out: dict = {"checks": check_ground_truth(gt, s1["entity_id"], s23_ids)}
    del s23_ids

    lists = gt["matched_entity_ids"]
    card = np.where(lists == "", 0, lists.str.count(",") + 1)
    n2 = lists.str.count(r"S2-").to_numpy()
    n3 = lists.str.count(r"S3-").to_numpy()
    g = pd.DataFrame({"s1_id": gt["source1_entity_id"], "card": card, "n2": n2, "n3": n3})
    g = g.merge(s1, left_on="s1_id", right_on="entity_id", how="left")
    out["cardinality"] = g["card"].value_counts().sort_index().to_dict()
    out["cardinality_by_country"] = {
        c: (x["card"].clip(upper=8).value_counts(normalize=True).sort_index().round(4).to_dict())
        for c, x in g.groupby("country")
    }
    out["singleton_rate"] = float((g["card"] == 0).mean())
    out["singleton_rate_by_country"] = g.groupby("country")["card"].apply(lambda x: float((x == 0).mean())).to_dict()
    out["mean_matches"] = float(g["card"].mean())
    out["mean_matches_nonsingleton"] = float(g.loc[g.card > 0, "card"].mean())
    out["mean_matches_by_country"] = g.groupby("country")["card"].mean().to_dict()
    out["n2_dist"] = pd.Series(n2).clip(upper=6).value_counts().sort_index().to_dict()
    out["n3_dist"] = pd.Series(n3).clip(upper=6).value_counts().sort_index().to_dict()
    out["source_mix"] = {
        "S2_and_S3": int(((n2 > 0) & (n3 > 0)).sum()),
        "S2_only": int(((n2 > 0) & (n3 == 0)).sum()),
        "S3_only": int(((n2 == 0) & (n3 > 0)).sum()),
        "none": int(((n2 == 0) & (n3 == 0)).sum()),
    }
    out["multi_within_same_source"] = float(((n2 >= 2) | (n3 >= 2)).mean())
    out["total_links"] = {"S2": int(n2.sum()), "S3": int(n3.sum())}

    # singleton rate vs S1 characteristics
    name_rep = s1["business_name"].map(s1["business_name"].value_counts()) > 1
    g["name_repeated"] = name_rep.values
    g["addr_has_postal"] = g["business_address"].str.contains(r"(?<!\d)\d{5,6}(?!\d)", regex=True).values
    out["singleton_rate_by_name_repeated"] = g.groupby("name_repeated")["card"].apply(lambda x: float((x == 0).mean())).to_dict()
    out["singleton_rate_by_addr_postal"] = g.groupby("addr_has_postal")["card"].apply(lambda x: float((x == 0).mean())).to_dict()

    # linkage from the S2/S3 side
    pairs = lists[lists != ""].str.split(",").explode()
    linked = set(pairs)
    for name, df in (("S2", s2), ("S3", s3)):
        lk = df["entity_id"].isin(linked)
        out[f"{name}_linked_rate"] = float(lk.mean())
        out[f"{name}_linked_rate_by_country"] = lk.groupby(df["country"]).mean().to_dict()

    # all-pairs space (same country only) vs positives
    c1 = s1["country"].value_counts()
    c23 = pd.concat([s2["country"], s3["country"]]).value_counts()
    same_country = int(sum(int(c1[c]) * int(c23.get(c, 0)) for c in c1.index))
    out["pair_space"] = {
        "cross_product_all": int(len(s1)) * int(len(s2) + len(s3)),
        "cross_product_same_country": same_country,
        "positives": int(len(pairs)),
        "neg_per_pos_same_country": same_country / max(1, len(pairs)),
    }
    return out


# --------------------------------------------------------------------------- pair agreement
_NUM = re.compile(r"\d+")
_POSTAL = re.compile(r"(?<!\d)\d{5,6}(?!\d)")


def _numbers(s: str) -> set:
    return {x.lstrip("0") or "0" for x in _NUM.findall(s)}


def _first_number(s: str) -> str | None:
    m = _NUM.search(s)
    return (m.group(0).lstrip("0") or "0") if m else None


def _postal(s: str) -> set:
    return set(_POSTAL.findall(s))


def pair_agreement() -> dict:
    """Compare true pairs vs random same-country pairs on simple agreement signals."""
    rng = np.random.default_rng(SEED)
    s1 = load_source("train", 1).set_index("entity_id")
    s23 = pd.concat([load_source("train", 2), load_source("train", 3)]).set_index("entity_id")
    gt = load_ground_truth()
    matched = gt[gt["matched_entity_ids"] != ""]
    samp = matched.iloc[rng.choice(len(matched), size=PAIR_SAMPLE_S1, replace=False)]
    pos = samp.set_index("source1_entity_id")["matched_entity_ids"].str.split(",").explode().reset_index()
    pos.columns = ["s1_id", "o_id"]
    pos["label"] = 1

    # random negatives: same country, uniform over S2+S3, one per positive
    pos_country = s1.loc[pos["s1_id"], "country"].to_numpy()
    neg_ids = np.empty(len(pos), dtype=object)
    s23_country_groups = s23.groupby("country").indices
    s23_index = s23.index.to_numpy()
    for c in np.unique(pos_country):
        m = pos_country == c
        neg_ids[m] = s23_index[rng.choice(s23_country_groups[c], size=int(m.sum()))]
    neg = pd.DataFrame({"s1_id": pos["s1_id"].to_numpy(), "o_id": neg_ids, "label": 0})
    pairs = pd.concat([pos, neg], ignore_index=True)

    a = s1.loc[pairs["s1_id"]]
    b = s23.loc[pairs["o_id"]]
    pairs["country"] = a["country"].to_numpy()
    pairs["source"] = pairs["o_id"].str[:2]
    pairs["country_equal"] = a["country"].to_numpy() == b["country"].to_numpy()

    an, bn = basic_normalize_series(a["business_name"]).to_numpy(), basic_normalize_series(b["business_name"]).to_numpy()
    aa, ba = basic_normalize_series(a["business_address"]).to_numpy(), basic_normalize_series(b["business_address"]).to_numpy()
    pairs["b_name_non_ascii"] = b["business_name"].str.contains(r"[^\x00-\x7f]").to_numpy()
    pairs["b_addr_empty"] = (b["business_address"].str.strip() == "").to_numpy()

    pairs["name_exact_norm"] = an == bn
    pairs["name_token_set"] = process.cpdist(an, bn, scorer=fuzz.token_set_ratio, workers=-1)
    pairs["name_ratio"] = process.cpdist(an, bn, scorer=fuzz.ratio, workers=-1)
    pairs["addr_token_set"] = process.cpdist(aa, ba, scorer=fuzz.token_set_ratio, workers=-1)

    # token overlap; "generic" tokens = 300 most frequent normalised name tokens in train
    name_tok = pd.concat([basic_normalize_series(s1["business_name"]),
                          basic_normalize_series(s23["business_name"].sample(2_000_000, random_state=SEED))])
    generic_counts = _token_counts(name_tok).head(300)
    generic = set(generic_counts.index)
    del name_tok
    shared_any, shared_rare, jacc = [], [], []
    for x, y in zip(an, bn):
        tx, ty = set(x.split()), set(y.split())
        inter = tx & ty
        shared_any.append(bool(inter))
        shared_rare.append(bool(inter - generic))
        jacc.append(len(inter) / len(tx | ty) if (tx or ty) else 0.0)
    pairs["name_share_token"] = shared_any
    pairs["name_share_rare_token"] = shared_rare
    pairs["name_jaccard"] = jacc

    ra, rb = a["business_address"].to_numpy(), b["business_address"].to_numpy()
    share_num, first_num, both_postal, postal_eq, addr_share_tok = [], [], [], [], []
    for x, y, xn, yn in zip(ra, rb, aa, ba):
        nx, ny = _numbers(x), _numbers(y)
        share_num.append(bool(nx & ny))
        f = _first_number(x)
        first_num.append(f is not None and f in ny)
        px, py = _postal(x), _postal(y)
        both_postal.append(bool(px) and bool(py))
        postal_eq.append(bool(px & py))
        addr_share_tok.append(bool(set(xn.split()) & set(yn.split())))
    pairs["addr_share_number"] = share_num
    pairs["addr_first_number_in_other"] = first_num
    pairs["both_have_postal"] = both_postal
    pairs["postal_equal"] = postal_eq
    pairs["addr_share_token"] = addr_share_tok
    pairs["no_rare_name_and_no_number"] = ~pairs["name_share_rare_token"] & ~pairs["addr_share_number"]

    bool_cols = ["country_equal", "name_exact_norm", "name_share_token", "name_share_rare_token",
                 "addr_share_token", "addr_share_number", "addr_first_number_in_other",
                 "both_have_postal", "postal_equal", "no_rare_name_and_no_number",
                 "b_name_non_ascii", "b_addr_empty"]
    num_cols = ["name_token_set", "name_ratio", "name_jaccard", "addr_token_set"]
    out = {
        "n_s1_sampled": PAIR_SAMPLE_S1,
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
        "rates_by_label": pairs.groupby("label")[bool_cols].mean().T.to_dict(),
        "rates_pos_by_country": pairs[pairs.label == 1].groupby("country")[bool_cols].mean().T.to_dict(),
        "rates_pos_by_source": pairs[pairs.label == 1].groupby("source")[bool_cols].mean().T.to_dict(),
        "numeric_median_by_label": pairs.groupby("label")[num_cols].median().T.to_dict(),
        "numeric_q10_pos": pairs[pairs.label == 1][num_cols].quantile(0.10).to_dict(),
        "numeric_q90_neg": pairs[pairs.label == 0][num_cols].quantile(0.90).to_dict(),
        "name_token_set_pos_hist": pd.cut(pairs.loc[pairs.label == 1, "name_token_set"], [-1, 20, 40, 60, 80, 99.99, 100])
                                     .value_counts(normalize=True, sort=False).round(4).astype(float)
                                     .rename(index=str).to_dict(),
        "addr_token_set_pos_hist": pd.cut(pairs.loc[pairs.label == 1, "addr_token_set"], [-1, 20, 40, 60, 80, 99.99, 100])
                                     .value_counts(normalize=True, sort=False).round(4).astype(float)
                                     .rename(index=str).to_dict(),
        "n_generic_tokens": len(generic),
        "generic_tokens_top40": list(generic_counts.index[:40]),
    }

    # hard examples: positives with weak name AND weak address evidence
    hard = pairs[(pairs.label == 1) & (pairs.name_token_set < 40)]
    out["weak_name_positive_rate"] = float(len(hard) / len(pos))
    out["weak_name_positive_addr_share_number"] = float(hard["addr_share_number"].mean()) if len(hard) else None
    ex = hard.sample(min(8, len(hard)), random_state=SEED)
    out["weak_name_examples"] = [
        [s1.at[r.s1_id, "business_name"], s23.at[r.o_id, "business_name"],
         s1.at[r.s1_id, "business_address"], s23.at[r.o_id, "business_address"]]
        for r in ex.itertuples()
    ]

    # exact-normalised-name rule: how precise is it on the full S2+S3 universe?
    samp_s1 = s1.loc[samp["source1_entity_id"]]
    left = pd.DataFrame({"s1_id": samp_s1.index, "country": samp_s1["country"].to_numpy(),
                         "nn": basic_normalize_series(samp_s1["business_name"]).to_numpy()})
    right = pd.DataFrame({"o_id": s23.index, "country": s23["country"].to_numpy(),
                          "nn": basic_normalize_series(s23["business_name"]).to_numpy()})
    right = right[right["nn"].isin(set(left["nn"]))]
    hits = left.merge(right, on=["country", "nn"])
    pos_set = set(zip(pos["s1_id"], pos["o_id"]))
    hits["is_pos"] = [(x, y) in pos_set for x, y in zip(hits["s1_id"], hits["o_id"])]
    out["exact_norm_name_rule"] = {
        "hits": int(len(hits)),
        "precision": float(hits["is_pos"].mean()) if len(hits) else None,
        "recall_of_positives": float(hits["is_pos"].sum() / len(pos)),
        "s1_with_any_hit": float(hits["s1_id"].nunique() / len(left)),
    }
    return out


# --------------------------------------------------------------------------- driver
def run_profile() -> dict:
    t0 = time.time()
    stats: dict = {"files": {}}
    for split in ("train", "test"):
        for k in (1, 2, 3):
            print(f"profiling {split} source{k} ...", flush=True)
            stats["files"][f"{split}_s{k}"] = profile_file(split, k)
    print("profiling ground truth ...", flush=True)
    stats["gt"] = profile_ground_truth()
    print("pair agreement analysis ...", flush=True)
    stats["pairs"] = pair_agreement()
    vm = psutil.virtual_memory()
    stats["system"] = {
        "total_ram_gb": vm.total / 1e9,
        "available_ram_gb": vm.available / 1e9,
        "cpu_logical": psutil.cpu_count(),
        "peak_rss_gb": psutil.Process().memory_info().peak_wset / 1e9
        if hasattr(psutil.Process().memory_info(), "peak_wset") else psutil.Process().memory_info().rss / 1e9,
        "profile_runtime_s": time.time() - t0,
    }
    return stats
