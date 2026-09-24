"""Command-line entry point for the business entity resolution pipeline.

Usage:
    python run_pipeline.py profile                # dataset forensics -> reports/dataset_profile.md
    python run_pipeline.py profile --render-only  # re-render the report from cached stats
    python run_pipeline.py baseline               # Stage-4 baseline (dev: 50k validation entities)
    python run_pipeline.py baseline --eval-n 0    # baseline on the full validation fold
"""
from __future__ import annotations

import argparse
import pickle
import sys

sys.path.insert(0, "src")

from business_entity_resolution.config import REPORTS_DIR, cache_path  # noqa: E402


def cmd_profile(args: argparse.Namespace) -> None:
    from business_entity_resolution.data.profile_report import render_profile

    stats_path = cache_path("profile", "stats.pkl")
    if not args.render_only:
        from business_entity_resolution.data.profiling import run_profile

        stats = run_profile()
        stats_path.write_bytes(pickle.dumps(stats))
    stats = pickle.loads(stats_path.read_bytes())
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / "dataset_profile.md"
    out.write_text(render_profile(stats), encoding="utf-8")
    print(f"wrote {out}")


def cmd_baseline(args: argparse.Namespace) -> None:
    from business_entity_resolution.pipeline.baseline_experiment import run_baseline

    eval_n = None if args.eval_n <= 0 else args.eval_n
    run_baseline(eval_n=eval_n, tune_n=args.tune_n, top_k=args.top_k, variants=tuple(args.variants), tag=args.tag)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("profile", help="dataset forensics report")
    p.add_argument("--render-only", action="store_true")
    p.set_defaults(func=cmd_profile)
    b = sub.add_parser("baseline", help="Stage-4 transparent baseline on the fold-0 validation split")
    b.add_argument("--eval-n", type=int, default=50_000, help="validation entities (<=0: full fold)")
    b.add_argument("--tune-n", type=int, default=50_000, help="training-fold entities used to tune (w, t)")
    b.add_argument("--top-k", type=int, default=50)
    b.add_argument("--variants", nargs="+", default=["B0_raw", "B1_norm"])
    b.add_argument("--tag", default="dev")
    b.set_defaults(func=cmd_baseline)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
