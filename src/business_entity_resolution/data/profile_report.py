"""Render the dataset forensics statistics (see ``profiling.py``) as Markdown."""
from __future__ import annotations

import datetime as _dt

import pandas as pd

FILES = ["train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"]


# --------------------------------------------------------------------------- formatting
def _fmt(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if v != v:
            return "–"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 10:
            return f"{v:.1f}"
        return f"{v:.3g}"
    return str(v).replace("|", "\\|").replace("\n", " ")


def _pct(v) -> str:
    return "–" if v is None or v != v else f"{100 * v:.2f}%"


def table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_fmt(x) for x in r) + " |")
    return "\n".join(out)


def df_table(df: pd.DataFrame, pct: bool = False, index_name: str = "") -> str:
    rows = [[i] + [(_pct(v) if pct else v) for v in r] for i, r in zip(df.index, df.values.tolist())]
    return table(rows, [index_name] + [str(c) for c in df.columns])


def _q_row(label: str, q: dict) -> list:
    return [label] + [q[k] for k in ("p1", "p5", "p25", "p50", "p75", "p95", "p99", "p100", "mean")]


Q_HEAD = ["", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max", "mean"]


# --------------------------------------------------------------------------- sections
def _overview(F: dict) -> str:
    rows = []
    for f in FILES:
        s = F[f]
        rows.append([s["file"], s["size_mb"], s["raw_lines"], len(s["dtypes"]), s["malformed_rows"],
                     s["rows_with_quote"], s["rows_with_cr"], s["bom"], s["parquet_mb"], s["mem_gb"]])
    t = table(rows, ["file", "TSV MB", "data rows", "cols", "malformed rows (≠4 fields)", "rows containing `\"`",
                     "rows with CR", "BOM", "Parquet MB", "in-memory GB"])
    checks = table([[f] + list(F[f]["frame_checks"].values()) for f in FILES],
                   ["file"] + list(F[FILES[0]]["frame_checks"].keys()))
    return (
        "## 1. Dataset overview\n\n"
        "All four columns are `entity_id`, `business_name`, `business_address`, `country`; every column is read "
        "as a string (`dtype=str`, `quoting=QUOTE_NONE`, `na_filter=False`).\n\n"
        f"{t}\n\n**Integrity checks on the loaded frames** (all should be 0 except `n`):\n\n{checks}\n"
    )


def _sources(F: dict) -> str:
    countries = sorted({c for f in FILES for c in F[f]["country_counts"]})
    rows = []
    for f in FILES:
        cc = F[f]["country_counts"]
        n = sum(cc.values())
        rows.append([f, n] + [f"{cc.get(c, 0):,} ({100 * cc.get(c, 0) / n:.1f}%)" for c in countries])
    ratio = [
        ["train", F["train_s2"]["raw_lines"] / F["train_s1"]["raw_lines"], F["train_s3"]["raw_lines"] / F["train_s1"]["raw_lines"]],
        ["test", F["test_s2"]["raw_lines"] / F["test_s1"]["raw_lines"], F["test_s3"]["raw_lines"] / F["test_s1"]["raw_lines"]],
    ]
    return (
        "## 2. Source statistics and country analysis\n\n"
        + table(rows, ["file", "rows"] + countries)
        + "\n\nRecords per S1 entity:\n\n"
        + table(ratio, ["split", "S2 / S1", "S3 / S1"])
        + "\n"
    )


def _missing(F: dict) -> str:
    rows = []
    for f in FILES:
        s = F[f]
        n = s["raw_lines"]
        ap = s["addr_patterns"]
        rows.append([f, s["name_empty"], s["addr_empty"], _pct(s["addr_empty"] / n), s["country_empty"],
                     _pct(ap.loc["null_literal", "ALL"])])
    by_c = pd.DataFrame({f: F[f]["addr_patterns"].loc["empty"] for f in FILES}).T
    return (
        "## 3. Missingness\n\n"
        + table(rows, ["file", "empty names", "empty addresses", "empty address %", "empty country",
                       "address contains NULL/N/A literal"])
        + "\n\nEmpty-address rate by country:\n\n" + df_table(by_c, pct=True, index_name="file") + "\n"
    )


def _duplicates(F: dict) -> str:
    keys = list(F["train_s1"]["dup"].keys())
    rows = [[f] + [F[f]["dup"][k] for k in keys] for f in FILES]
    out = "## 4. Duplicate analysis\n\n" + table(rows, ["file"] + keys) + "\n\n"
    out += "Most frequent exact business names (count):\n\n"
    for f in FILES:
        out += f"- **{f}**: " + ", ".join(f"`{k}` ({v})" for k, v in list(F[f]["top_names"].items())[:8]) + "\n"
    out += "\nMost frequent exact addresses (count):\n\n"
    for f in FILES:
        out += f"- **{f}**: " + ", ".join(f"`{k}` ({v})" for k, v in list(F[f]["top_addresses"].items())[:4]) + "\n"
    return out


def _names(F: dict) -> str:
    out = "## 5. Name statistics\n\n### Length\n\nCharacters:\n\n"
    out += table([_q_row(f, F[f]["name_chars"]) for f in FILES], Q_HEAD)
    out += "\n\nWhitespace tokens:\n\n" + table([_q_row(f, F[f]["name_tokens"]) for f in FILES], Q_HEAD)
    out += "\n\nMedian name length (chars) by country:\n\n" + table(
        [[f] + [f"{c}: {v:.0f}" for c, v in F[f]["median_len_by_country"]["nl"].items()] for f in FILES], ["file", "medians"])
    out += "\n\n### Casing\n\n" + table(
        [[f, _pct(F[f]["name_case"]["all_upper"]), _pct(F[f]["name_case"]["all_lower"]), _pct(F[f]["name_case"]["mixed"])] for f in FILES],
        ["file", "ALL UPPER", "all lower", "mixed"])
    pat = pd.DataFrame({f: F[f]["name_patterns"]["ALL"] for f in FILES})
    out += "\n\n### Noise patterns (share of rows, all countries)\n\n" + df_table(pat, pct=True, index_name="pattern")
    for f in ("train_s2", "test_s2"):
        out += f"\n\nBy country — **{f}**:\n\n" + df_table(F[f]["name_patterns"], pct=True, index_name="pattern")
    out += "\n\n### Unicode scripts in names (share of rows containing ≥1 char of the script; sampled 200k rows/file)\n\n"
    for f in FILES:
        sc = F[f]["business_name_scripts_sampled"]
        out += f"- **{f}**: " + (", ".join(f"{k} {_pct(v)}" for k, v in sc.items()) or "none") + "\n"
    out += "\nMost frequent non-ASCII characters in names (sampled):\n\n"
    for f in FILES:
        ch = F[f]["business_name_nonascii_chars_sampled"]
        out += f"- **{f}**: " + (" ".join(f"`{k}`×{v}" for k, v in list(ch.items())[:20]) or "none") + "\n"
    out += "\nASCII punctuation in names (sampled counts):\n\n"
    for f in FILES:
        out += f"- **{f}**: " + " ".join(f"`{k}`×{v}" for k, v in F[f]["name_ascii_punct_sampled"].items()) + "\n"
    out += "\n### Normalised name tokens\n\nVocabulary size (distinct normalised tokens) by country:\n\n"
    out += table([[f] + [f"{c}: {v:,}" for c, v in F[f]["name_vocab_size"].items()] for f in FILES], ["file", "vocab"])
    out += "\n\nTop tokens by document frequency (share of names containing token) — per country:\n\n"
    for f in ("train_s1", "train_s2", "test_s1", "test_s2"):
        for c, d in F[f]["top_name_tokens_df"].items():
            out += f"- **{f} / {c}**: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in list(d.items())[:20]) + "\n"
    out += "\nLast name token (legal-suffix proxy), share of names:\n\n"
    for f in ("train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"):
        for c, d in F[f]["top_last_name_token"].items():
            out += f"- **{f} / {c}**: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in list(d.items())[:14]) + "\n"
    return out


def _addresses(F: dict) -> str:
    out = "## 6. Address statistics\n\n### Length (non-empty addresses)\n\nCharacters:\n\n"
    out += table([_q_row(f, F[f]["addr_chars"]) for f in FILES], Q_HEAD)
    out += "\n\nWhitespace tokens:\n\n" + table([_q_row(f, F[f]["addr_tokens"]) for f in FILES], Q_HEAD)
    out += "\n\nComma-separated components:\n\n" + table([_q_row(f, F[f]["addr_components"]) for f in FILES], Q_HEAD)
    out += "\n\nMedian address length (chars) by country:\n\n" + table(
        [[f] + [f"{c}: {v:.0f}" for c, v in F[f]["median_len_by_country"]["al"].items()] for f in FILES], ["file", "medians"])
    out += "\n\n### Address patterns by country\n\n"
    for f in FILES:
        out += f"**{f}**\n\n" + df_table(F[f]["addr_patterns"], pct=True, index_name="pattern") + "\n\n"
    out += "### Unicode scripts in addresses (sampled 200k rows/file)\n\n"
    for f in FILES:
        sc = F[f]["business_address_scripts_sampled"]
        out += f"- **{f}**: " + (", ".join(f"{k} {_pct(v)}" for k, v in sc.items()) or "none") + "\n"
    out += "\n### Last address component (typically state / region), share of rows\n\n"
    for f in ("train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"):
        for c, d in F[f]["top_last_addr_component"].items():
            out += f"- **{f} / {c}**: " + ", ".join(f"`{k}` {100 * v:.1f}%" for k, v in list(d.items())[:10]) + "\n"
    return out


def _examples(F: dict) -> str:
    out = "## 7. Raw examples (random, seeded)\n\n"
    for f in FILES:
        out += f"**{f}**\n\n"
        rows = [r for c in sorted(F[f]["examples"]) for r in F[f]["examples"][c][:3]]
        out += table(rows, ["entity_id", "business_name", "business_address"]) + "\n\n"
    return out


def _ground_truth(G: dict) -> str:
    out = "## 8. Ground-truth statistics\n\n**Integrity:**\n\n" + table([[k, v] for k, v in G["checks"].items()], ["check", "value"])
    tot = sum(G["cardinality"].values())
    out += "\n\n### Match cardinality (number of S2+S3 matches per S1 entity)\n\n"
    out += table([[k, v, _pct(v / tot)] for k, v in G["cardinality"].items()], ["matches", "S1 entities", "share"])
    out += f"\n\n- Mean matches per S1: **{G['mean_matches']:.3f}**; among non-singletons: **{G['mean_matches_nonsingleton']:.3f}**\n"
    out += "- Mean matches by country: " + ", ".join(f"{c} {v:.3f}" for c, v in G["mean_matches_by_country"].items()) + "\n"
    out += f"- Total positive links: S2 {G['total_links']['S2']:,}, S3 {G['total_links']['S3']:,}\n"
    out += f"- S1 entities with ≥2 matches **within the same source** (S2 or S3 duplicates): {_pct(G['multi_within_same_source'])}\n"
    out += "\nCardinality distribution by country (8 = 8+):\n\n"
    cc = pd.DataFrame(G["cardinality_by_country"]).fillna(0.0)
    out += df_table(cc, pct=True, index_name="matches")
    out += "\n\nMatches per S1 from each source (6 = 6+):\n\n"
    idx = sorted(set(G["n2_dist"]) | set(G["n3_dist"]))
    out += table([[i, G["n2_dist"].get(i, 0), G["n3_dist"].get(i, 0)] for i in idx], ["count", "S1 with this many S2 matches", "… S3 matches"])
    out += "\n\nSource mix of matches:\n\n" + table([[k, v, _pct(v / tot)] for k, v in G["source_mix"].items()], ["mix", "S1 entities", "share"])
    out += "\n\n### Singleton statistics\n\n"
    out += f"- Overall singleton rate: **{_pct(G['singleton_rate'])}**\n"
    out += "- By country: " + ", ".join(f"{c} {_pct(v)}" for c, v in G["singleton_rate_by_country"].items()) + "\n"
    out += "- By whether the S1 exact name is repeated elsewhere in S1: " + ", ".join(
        f"repeated={k}: {_pct(v)}" for k, v in G["singleton_rate_by_name_repeated"].items()) + "\n"
    out += "- By whether the S1 address contains a 5–6 digit token: " + ", ".join(
        f"has_postal={k}: {_pct(v)}" for k, v in G["singleton_rate_by_addr_postal"].items()) + "\n"
    out += "\n### Linkage from the S2/S3 side\n\n"
    for s in ("S2", "S3"):
        out += f"- {s} records linked to some S1: **{_pct(G[f'{s}_linked_rate'])}** (" + ", ".join(
            f"{c} {_pct(v)}" for c, v in G[f"{s}_linked_rate_by_country"].items()) + ")\n"
    ps = G["pair_space"]
    out += (f"\n### Pair space\n\n- Full cross product S1 × (S2+S3): {ps['cross_product_all']:,}\n"
            f"- Same-country cross product: {ps['cross_product_same_country']:,}\n"
            f"- Positive pairs: {ps['positives']:,}\n"
            f"- Negatives per positive (same-country all-pairs): **{ps['neg_per_pos_same_country']:,.0f} : 1**\n")
    return out


def _pairs(P: dict) -> str:
    out = ("## 9. Agreement between true matches (positives) vs random same-country pairs\n\n"
           f"Sample: {P['n_s1_sampled']:,} non-singleton train S1 entities → {P['n_pos']:,} positive pairs, "
           f"and the same number of random same-country S1–(S2∪S3) pairs as a baseline. Names/addresses use the "
           "forensic normaliser (NFKC → anyascii transliteration → lowercase → non-alphanumerics to space). "
           f"\"Rare\" token = not among the {P['n_generic_tokens']} most frequent normalised name tokens in train.\n\n")
    r = pd.DataFrame(P["rates_by_label"]).rename(columns={0: "random pairs", 1: "true matches"})
    out += df_table(r[["true matches", "random pairs"]], pct=True, index_name="signal")
    out += "\n\nPositives by country:\n\n" + df_table(pd.DataFrame(P["rates_pos_by_country"]), pct=True, index_name="signal")
    out += "\n\nPositives by source:\n\n" + df_table(pd.DataFrame(P["rates_pos_by_source"]), pct=True, index_name="signal")
    m = pd.DataFrame(P["numeric_median_by_label"]).rename(columns={0: "random median", 1: "true median"})
    m["true p10"] = pd.Series(P["numeric_q10_pos"])
    m["random p90"] = pd.Series(P["numeric_q90_neg"])
    out += "\n\nSimilarity scores (0–100; Jaccard 0–1):\n\n" + df_table(m[["true median", "true p10", "random median", "random p90"]], index_name="score")
    out += "\n\nDistribution of name token-set similarity among true matches:\n\n" + table(
        [[k, _pct(v)] for k, v in P["name_token_set_pos_hist"].items()], ["bin", "share"])
    out += "\n\nDistribution of address token-set similarity among true matches:\n\n" + table(
        [[k, _pct(v)] for k, v in P["addr_token_set_pos_hist"].items()], ["bin", "share"])
    e = P["exact_norm_name_rule"]
    out += ("\n\n**Exact normalised-name rule** (sampled S1 vs. the entire train S2∪S3, same country): "
            f"{e['hits']:,} hits, precision **{_pct(e['precision'])}**, recall of positives **{_pct(e['recall_of_positives'])}**, "
            f"S1 with ≥1 hit {_pct(e['s1_with_any_hit'])}.\n")
    out += (f"\n**Weak-name positives** (name token-set < 40): {_pct(P['weak_name_positive_rate'])} of positives; "
            f"of these, {_pct(P['weak_name_positive_addr_share_number'])} share an address number with the S1 record. Examples:\n\n")
    out += table(P["weak_name_examples"], ["S1 name", "match name", "S1 address", "match address"])
    out += "\n\nMost frequent normalised name tokens (treated as generic): " + ", ".join(f"`{t}`" for t in P["generic_tokens_top40"]) + "\n"
    return out


def _compute(F: dict, S: dict) -> str:
    rows = [[f, F[f]["size_mb"], F[f]["raw_scan_s"], F[f]["load_s"], F[f]["mem_gb"]] for f in FILES]
    tr = sum(F[f]["mem_gb"] for f in FILES[:3])
    te = sum(F[f]["mem_gb"] for f in FILES[3:])
    return (
        "## 10. Computational / memory assessment\n\n"
        f"Machine: {S['total_ram_gb']:.1f} GB RAM, {S['cpu_logical']} logical CPUs. "
        f"Profiling peak working set: {S['peak_rss_gb']:.2f} GB; profiling runtime {S['profile_runtime_s'] / 60:.1f} min.\n\n"
        + table(rows, ["file", "TSV MB", "raw stream scan s", "load s (Parquet if cached)", "in-memory GB (Arrow strings)"])
        + f"\n\n- All three train sources together: **{tr:.2f} GB** in memory; all test sources: **{te:.2f} GB**.\n"
    )


def render_profile(stats: dict) -> str:
    F, G, P, S = stats["files"], stats["gt"], stats["pairs"], stats["system"]
    from .profile_observations import observations as obs  # narrative bullets computed from stats

    parts = [
        "# Dataset forensics report\n",
        f"_Generated {_dt.date.today().isoformat()} by `python run_pipeline.py profile` "
        "(`src/business_entity_resolution/data/profiling.py`). Every number below is computed from the "
        "supplied files; sampled statistics are labelled as such._\n",
        _overview(F), _sources(F), _missing(F), _duplicates(F), _names(F), _addresses(F), _examples(F),
        _ground_truth(G), _pairs(P), _compute(F, S), obs(stats),
    ]
    return "\n".join(parts)
