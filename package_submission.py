"""Build the final submission zip in the structure required by the challenge:

<team>_submission.zip
├── output/matching_results.tsv, candidate_pairs.tsv
├── code/business_entity_resolution/{src/, README.md, requirements.txt, run_pipeline.py,
│                                    pyproject.toml, utils/, tests/, models/, experiments/, reports/}
└── Documentation_template.md   (filled-in methodology)

Usage:  python package_submission.py --team <team_name>
Refuses to package unless the official validator passes on output/.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE_ITEMS = ["src", "tests", "utils", "models", "experiments", "reports", "notebooks",
              "run_pipeline.py", "package_submission.py", "pyproject.toml", "requirements.txt", "README.md"]
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints"}
SKIP_SUFFIX = {".pyc"}


def validate() -> None:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    cmd = [sys.executable, str(ROOT / "utils" / "validate_submission.py"),
           "--matching", str(ROOT / "output" / "matching_results.tsv"),
           "--candidate", str(ROOT / "output" / "candidate_pairs.tsv"),
           "--test-dir", str(ROOT / "dataset" / "test")]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    print(res.stdout)
    if res.returncode != 0:
        sys.exit("validator FAILED - not packaging")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()
    if not args.skip_validate:
        validate()
    out = ROOT / f"{args.team}_submission.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(ROOT / "output" / f, f"output/{f}")
        for item in CODE_ITEMS:
            p = ROOT / item
            if not p.exists():
                continue
            files = [p] if p.is_file() else [x for x in p.rglob("*") if x.is_file()]
            for x in files:
                rel = x.relative_to(ROOT)
                if SKIP_PARTS & set(rel.parts) or x.suffix in SKIP_SUFFIX or rel.parts[0] == "output":
                    continue
                z.write(x, f"code/business_entity_resolution/{rel.as_posix()}")
        z.write(ROOT / "Documentation_template.md", "Documentation_template.md")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
