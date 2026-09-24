# Business Entity Resolution — ML Challenge 2026

Pipeline to match every Source-1 business entity to its Source-2 / Source-3 records,
optimised for macro-averaged F₀.₅. Runs fully offline on the supplied data only
(no external lookups, geocoding or enrichment).

> Status: **Stage 2 (dataset forensics) complete.** The remaining sections are filled in as each
> stage lands.
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

## 4–10. Training, validation, inference, submission validation

_To be added with the corresponding pipeline stages._

Submission format check (official validator, stdlib only):

```bash
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Tests

```bash
pytest -q
```
