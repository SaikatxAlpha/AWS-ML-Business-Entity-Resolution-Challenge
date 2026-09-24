"""Command-line entry point for the business entity resolution pipeline.

Usage:
    python run_pipeline.py profile                # dataset forensics -> reports/dataset_profile.md
    python run_pipeline.py profile --render-only  # re-render the report from cached stats
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("profile", help="dataset forensics report")
    p.add_argument("--render-only", action="store_true")
    p.set_defaults(func=cmd_profile)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
