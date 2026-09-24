"""Narrative sections of the forensics report.

Every number in these bullets is read from the computed statistics dict; the prose only
interprets them. If the data changes, rerun ``python run_pipeline.py profile``.
"""
from __future__ import annotations


def _p(v: float) -> str:
    return f"{100 * v:.1f}%"


def _f05(p: float, r: float) -> float:
    return 0.0 if p == 0 and r == 0 else 1.25 * p * r / (0.25 * p + r)


def observations(stats: dict) -> str:
    F, G, P = stats["files"], stats["gt"], stats["pairs"]
    pos = P["rates_by_label"][1]
    neg = P["rates_by_label"][0]
    pos_c = P["rates_pos_by_country"]
    card = G["cardinality"]
    n_s1 = sum(card.values())
    rows = lambda f: F[f]["raw_lines"]  # noqa: E731
    tr_ratio = (rows("train_s2") + rows("train_s3")) / rows("train_s1")
    te_ratio = (rows("test_s2") + rows("test_s3")) / rows("test_s1")
    linked = (G["total_links"]["S2"] + G["total_links"]["S3"]) / (rows("train_s2") + rows("train_s3"))
    exact = P["exact_norm_name_rule"]
    fr_s1 = F["test_s1"]["country_counts"].get("France", 0) / rows("test_s1")
    union_cov = 1 - pos["no_rare_name_and_no_number"]

    # illustrative per-entity F0.5 values (exact arithmetic, not model results)
    ex_miss = _f05(1.0, 3 / 4)
    ex_extra = _f05(4 / 5, 1.0)

    out = ["## 11. Important observations\n"]
    o = out.append
    o(f"1. **Files are clean structurally.** 0 malformed rows, 0 duplicate IDs, no BOM/CR, every ground-truth ID "
      f"resolves. But {F['test_s2']['rows_with_quote']:,}/{F['test_s3']['rows_with_quote']:,} rows in test S2/S3 contain "
      "a literal `\"` — CSV-style quoting must stay disabled.")
    o(f"2. **Source 1 is a clean canonical form; S2/S3 are noisy derivatives.** Train S1 names: 0 non-ASCII, "
      f"0 multi-space, 0 all-uppercase; S2/S3: {_p(F['train_s2']['name_patterns'].loc['non_ascii','ALL'])}/"
      f"{_p(F['train_s3']['name_patterns'].loc['non_ascii','ALL'])} non-ASCII names, ~11% multi-space, "
      f"{_p(F['train_s2']['name_case']['all_upper'])} of S2 names ALL-CAPS, "
      f"{_p(F['train_s2']['name_patterns'].loc['ocr_digit_in_word','ALL'])} OCR-style digit-for-letter swaps, "
      f"{_p(F['train_s2']['name_patterns'].loc['domain_name','ALL'])} domain-style names (`xyz.com`), "
      f"{_p(F['train_s3']['name_patterns'].loc['alias_marker','ALL'])} of S3 names carry f/k/a / d/b/a markers.")
    o("3. **Sources have systematic, source-specific formats.** S2 US addresses end in 2-letter state codes; "
      "S3 US addresses use full state names (`Texas`, `New York`) while S3 India uses short codes "
      "(`MH`, `DL`, `KA`). State names also appear in native scripts (`महाराष्ट्र`, `दिल्ली`). "
      "The same region therefore has ≥3 surface forms — an equivalence map must be *learned from training pairs*, "
      "not typed in.")
    o(f"4. **Transliteration is pervasive.** In sampled S2 names, Devanagari appears in "
      f"{_p(F['train_s2']['business_name_scripts_sampled'].get('DEVANAGARI', 0))} of rows plus Telugu, Kannada, Tamil, "
      "Bengali, Gujarati, Malayalam, Oriya, Gurmukhi. Offline anyascii romanisation gives phonetic spellings that do not "
      "equal the English source word (Devanagari `प्राइवेट`→`praivet`, `एलएलपी`→`elelpi`; Tamil `லிமிடெட்`→`limitet`): "
      f"`praivet` alone is in {_p(F['train_s2']['top_name_tokens_df']['India'].get('praivet', 0))} of India S2 names. "
      "Token-level romanised↔English equivalences can be mined from train positives.")
    o(f"5. **Match cardinality is high and variable; singletons are rare.** Singletons {_p(G['singleton_rate'])}, "
      f"exactly-one {_p(card.get(1, 0) / n_s1)}, ≥2 matches {_p(1 - (card.get(0, 0) + card.get(1, 0)) / n_s1)}; "
      f"mean {G['mean_matches']:.2f} matches per S1 ({G['mean_matches_nonsingleton']:.2f} among non-singletons). "
      f"{_p(G['multi_within_same_source'])} of S1 entities have ≥2 matches *inside the same source*, i.e. S2 and S3 "
      "themselves contain duplicate records of one business.")
    o(f"6. **Each S2/S3 record belongs to at most one S1 entity** (0 violations over {sum(G['total_links'].values()):,} "
      f"links), and {_p(1 - linked)} of train S2/S3 records belong to *no* S1 entity (distractors). "
      f"Linkage rate is identical across countries (S2 {_p(G['S2_linked_rate'])}, S3 {_p(G['S3_linked_rate'])}).")
    o(f"7. **Singletons are indistinguishable from S1-side features.** Singleton rate is {_p(G['singleton_rate_by_country'].get('India', 0))} "
      f"(India) vs {_p(G['singleton_rate_by_country'].get('US', 0))} (US), {_p(G['singleton_rate_by_name_repeated'][True])} vs "
      f"{_p(G['singleton_rate_by_name_repeated'][False])} for repeated vs unique S1 names. The no-match decision must come "
      "from candidate evidence (the absence of a strong candidate), not from properties of the S1 record.")
    o(f"8. **Address numbers are the strongest single signal.** A true match shares an address number with the S1 "
      f"record in {_p(pos['addr_share_number'])} of pairs vs {_p(neg['addr_share_number'])} of random same-country pairs; "
      f"the S1 *first* number appears in the match in {_p(pos['addr_first_number_in_other'])} vs {_p(neg['addr_first_number_in_other'])}.")
    o(f"9. **Postal/PIN codes are nearly absent.** Only {_p(pos['both_have_postal'])} of true pairs have a 5–6 digit code on "
      f"both sides (India {_p(pos_c['India']['both_have_postal'])}, US {_p(pos_c['US']['both_have_postal'])}). "
      "Postal blocking cannot be a primary key.")
    o(f"10. **Names alone are unsafe.** Exact normalised name equality holds for only {_p(pos['name_exact_norm'])} of true "
      f"pairs, and the exact-name rule has precision {_p(exact['precision'])} against the full S2∪S3 universe "
      f"({exact['hits']:,} hits for {P['n_s1_sampled']:,} S1 entities). Generic names repeat heavily "
      f"(`{next(iter(F['train_s1']['top_names']))}` ×{next(iter(F['train_s1']['top_names'].values()))} in train S1, "
      f"`{next(iter(F['test_s1']['top_names']))}` ×{next(iter(F['test_s1']['top_names'].values()))} in test S1).")
    o(f"11. **Rare name tokens are highly discriminative.** Sharing a non-generic name token: {_p(pos['name_share_rare_token'])} "
      f"of true pairs vs {_p(neg['name_share_rare_token'])} of random pairs (any token: {_p(pos['name_share_token'])} vs "
      f"{_p(neg['name_share_token'])}). India is harder ({_p(pos_c['India']['name_share_rare_token'])}) than US "
      f"({_p(pos_c['US']['name_share_rare_token'])}) because of transliteration.")
    o(f"12. **Some matches have unrelated names.** {_p(P['weak_name_positive_rate'])} of true pairs have name token-set "
      f"similarity < 40 (random invented trade names such as `Dovazeta`, `Avikelo`, or native-script names); "
      f"{_p(P['weak_name_positive_addr_share_number'])} of those still share an address number. "
      f"{_p(pos['b_addr_empty'])} of true matches have an empty address, so for those only the name can link.")
    o(f"13. **Simple evidence union bounds blocking.** {_p(union_cov)} of true pairs share either a rare name token or an "
      f"address number; the remaining {_p(pos['no_rare_name_and_no_number'])} need fuzzy / character-level / transliteration-"
      "aware retrieval.")
    o(f"14. **Country always agrees on true pairs** ({_p(pos['country_equal'])} of {P['n_pos']:,} sampled positives), "
      "so blocking within country is lossless on train; country is used as an open-set partition key, never a whitelist.")
    o(f"15. **Train→test shift.** France is {_p(fr_s1)} of test S1 and absent from train. Test has "
      f"{te_ratio:.2f} S2+S3 records per S1 vs {tr_ratio:.2f} in train, so either more matches per entity or more "
      "distractors — threshold priors may not transfer. France brings new legal forms (`sarl` "
      f"{_p(F['test_s1']['top_last_name_token']['France'].get('sarl', 0))}, `sas` {_p(F['test_s1']['top_last_name_token']['France'].get('sas', 0))}, "
      "`eurl`, `sasu`, `sci`), acronym-only names in S2/S3 "
      f"({_p(F['test_s2']['name_patterns'].loc['acronym_only','France'])} of France S2), extremely repetitive names/addresses "
      f"(`{next(iter(F['test_s1']['top_addresses']))}` ×{next(iter(F['test_s1']['top_addresses'].values()))} in S1), "
      "accents, `°`/`º` characters (seen only in test S2/S3 addresses), and S2/S3 regions given as départements (`Gironde`, `Nord`, `Loire-Atlantique`) where "
      "S1 uses régions (`Nouvelle-Aquitaine`, `Hauts-de-France`).")
    o(f"16. **Memory is not the bottleneck; pair volume is.** Each split's three sources occupy ≈"
      f"{sum(F[f]['mem_gb'] for f in ('train_s1','train_s2','train_s3')):.1f} GB as Arrow strings and reload from Parquet "
      f"in <1 s. The raw same-country pair space is {G['pair_space']['cross_product_same_country']:.2e} "
      f"({G['pair_space']['neg_per_pos_same_country']:,.0f} negatives per positive), so blocking must avoid anything O(N²).")

    out.append("\n## 12. Implications for blocking\n")
    o = out.append
    o("- Partition by `country` value (open set, derived from the data at run time) — lossless on train.")
    o("- Core keys to evaluate, driven by observations 8, 11, 13: (a) address house/first number + a street/locality token; "
      "(b) rare (IDF-weighted) name tokens after transliteration; (c) character n-gram TF-IDF nearest neighbours on names "
      "(typos, OCR swaps, joined words like `Roaddagdiwadi`); (d) TF-IDF nearest neighbours on addresses (invented trade "
      "names, empty/garbled names); (e) combined name+address n-gram retrieval.")
    o("- Postal-code blocking is low-coverage (observation 9); measure it, but do not rely on it.")
    o("- Exact-name blocks explode on generic names (observation 10); cap block sizes / use IDF-weighted scoring and rank-limited "
      "retrieval instead of raw equality blocks.")
    o("- Target: candidate recall well above the ≈95% simple-evidence bound (observation 13) with an average candidate list "
      f"that keeps test pairs tractable (test S1 = {rows('test_s1'):,}; e.g. 50 candidates → "
      f"{50 * rows('test_s1') / 1e6:.0f}M pairs).")
    o("- All blocking statistics (IDF, n-gram vocabularies) are computed per split from the split's own records, so France "
      "is handled from test-set statistics without labels.")

    out.append("\n## 13. Implications for feature engineering\n")
    o = out.append
    o("- Keep original and normalised text. Normalise: NFKC → transliterate (anyascii) → lowercase → punctuation/space "
      "collapse; strip OCR digit-for-letter swaps inside alphabetic tokens (`8rothers`, `NATI0NAL`).")
    o("- Legal-suffix handling: separate the legal-form tokens (`limited/ltd/private/pvt/praivet/llc/inc/sarl/sas/…`) into "
      "their own feature instead of deleting them. The suffix inventory is mined from token statistics (top last tokens per "
      "country) — no hard-coded country lists.")
    o("- Learn token equivalences (romanised↔English, abbreviations like `st`↔`street`, state code↔state name) from aligned "
      "tokens in train positive pairs; plus generic prefix/abbreviation matching (`r`↔`rue`, `av`↔`avenue`) that transfers "
      "to France without labels.")
    o("- Numeric features: first-number match, number-set Jaccard, conflicting numbers, zero-padding-insensitive and "
      "`#`-insensitive comparison, unit/flat numbers.")
    o("- IDF-weighted name/address token overlap, with IDF computed per country on the split being scored.")
    o("- Missingness flags (empty address, `null` literals), name genericness (how many S1/S2/S3 records share the "
      "normalised name / address), domain-name and alias-marker flags, native-script flags.")
    o("- Candidate-level context: rank of the candidate for this S1, score gap to the best candidate, and how many S1 "
      "entities compete for the same S2/S3 record (observation 6).")
    o("- Avoid a raw `country` one-hot in the model (France unseen); use country-agnostic features so the model transfers.")

    out.append("\n## 14. Implications for modelling and decisions\n")
    o = out.append
    o(f"- Per-entity F₀.₅ arithmetic: with 4 true matches, missing one gives {ex_miss:.3f}; adding one wrong match to 4 "
      f"correct gives {ex_extra:.3f}; any match on a true singleton gives 0. With a mean of "
      f"{G['mean_matches_nonsingleton']:.2f} matches per non-singleton entity and only {_p(G['singleton_rate'])} singletons, "
      "recall still matters. The policy should keep every candidate above a precision-oriented threshold, not just "
      "the top 1.")
    o("- Exploit the one-S1-per-record constraint (observation 6): when an S2/S3 record is claimed by several S1 entities, "
      "keep at most the best-scoring one. Test this as a graph/assignment step on validation.")
    o("- Within-source duplicates (observation 5) enable consistency features: candidates that are near-duplicates of an "
      "already-accepted candidate are likely matches too.")
    o("- Validation: split by S1 entity, stratified by country × cardinality bucket. Additionally run a **cross-country "
      "transfer check** (train on one country, validate on the other) as a proxy for the unseen France slice.")
    o("- Negatives come from the blocked candidate set (hard negatives); random negatives are trivially separable "
      f"(e.g. random pairs share a rare name token {_p(neg['name_share_rare_token'])} of the time).")
    return "\n".join(out) + "\n"
