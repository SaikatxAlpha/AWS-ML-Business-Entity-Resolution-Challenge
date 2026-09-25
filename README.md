# Business Entity Resolution — ML Challenge 2026

Pipeline to match every Source-1 business entity to its Source-2 / Source-3 records,
optimised for macro-averaged F₀.₅. Runs fully offline on the supplied data only
(no external lookups, geocoding or enrichment).

> Status: **Stages 1–4 complete** (forensics, evaluator, transparent baseline, end-to-end test
> inference). Learned matcher (Stages 5+) in progress.
> The organisers' original problem statement is kept verbatim in
> [`docs/challenge_README.md`](docs/challenge_README.md).

## 1. Environment setup

Tested on Windows 11, Python 3.11.9, 16 GB RAM.

```bash
python3.11 -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps            # makes `business_entity_resolution` importable
```

All dependencies are permissively licensed (BSD / MIT / Apache-2.0 / ISC). Transliteration
uses `anyascii` (ISC) rather than `Unidecode` (GPL).

## 2. Dataset placement

```
dataset/
├── train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
└── test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
```

Override with `BER_DATA_DIR=/path/to/dataset`. Raw files are never modified; parsed copies
are cached as Parquet under `cache/` (override with `BER_CACHE_DIR`).

The files contain literal `"` characters, so every reader uses `quoting=QUOTE_NONE` and
`na_filter=False` (see `src/business_entity_resolution/data/schema.py`).

## 3. Dataset profiling

```bash
python run_pipeline.py profile               # ~20 min first run (builds Parquet cache)
python run_pipeline.py profile --render-only # re-render from cached statistics
```

Output: `reports/dataset_profile.md` (+ `notebooks/01_dataset_forensics.ipynb`).

## 4. Validation design (leakage-safe)

`src/business_entity_resolution/evaluation/validation.py`

- 5 folds over **Source-1 entities**, stratified by `country × match-count bucket (0..6+)`;
  fold 0 is the validation fold (441,370 entities). Cached in `cache/splits/folds_v1.parquet`
  (deterministic from `config.SEED`).
- Validation entities are matched against the **entire** train S2/S3 pool, as test entities
  are matched against the entire test pool. Inference code never receives labels.
- Everything label-dependent (token maps, weak tokens, thresholds, models) is fit on
  training-fold entities only (`models/normalizer_f0.json`). The final artefacts use all
  training labels (`models/normalizer_all.json`).
- `country_holdout(<country>)` provides a train-on-one-country split as a proxy for the
  unseen test country.
- Metric: `evaluation/metrics.py` is the exact competition F₀.₅, macro over every entity
  including singletons. Tests check it against the README worked example (0.714).

## 5. Baseline (Stage 4)

```bash
python run_pipeline.py baseline                  # dev: 50k val entities, (w,t) tuned on 50k train-fold entities
python run_pipeline.py baseline --eval-n 0       # full validation fold
```

Every run appends to `experiments/results.csv` and writes `experiments/runs/*.json`.
The rule with the best **training-fold** tuning score is written to
`models/baseline_rule.json`. The validation TSVs are written, read back and re-scored to
check the output files.

## 6. Test inference and submission validation

```bash
python run_pipeline.py infer      # normalise -> block -> candidate_pairs -> decide -> matching_results -> validator
```

Outputs: `output/matching_results.tsv`, `output/candidate_pairs.tsv`, `output/run_report.json`.
The official validator (`utils/validate_submission.py --check-ids`) runs automatically. To
run it by hand:

```bash
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

## Tests

```bash
pytest -q
```
