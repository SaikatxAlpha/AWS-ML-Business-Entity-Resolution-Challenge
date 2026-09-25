"""Command-line entry point for the business entity resolution pipeline.

Usage:
    python run_pipeline.py profile                # dataset forensics -> reports/dataset_profile.md
    python run_pipeline.py profile --render-only  # re-render the report from cached stats
    python run_pipeline.py baseline               # Stage-4 baseline (dev: 50k validation entities)
    python run_pipeline.py baseline --eval-n 0    # baseline on the full validation fold
    python run_pipeline.py train                  # Stages 5-18: blocking, features, XGBoost/LightGBM, policy, ablation
    python run_pipeline.py infer                  # test inference (models/matcher_spec.json) + official validator
    python run_pipeline.py infer --model baseline_rule.json   # baseline submission
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

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


def cmd_train(args: argparse.Namespace) -> None:
    from business_entity_resolution.pipeline.train_pipeline import (blocking_analysis, build_dev_features,
                                                                     evaluate_full_validation, train_and_select)

    if "build" in args.steps:
        build_dev_features()
    if "blocking" in args.steps:
        blocking_analysis()
    if "model" in args.steps:
        train_and_select(ablation=not args.no_ablation)
    if "fullval" in args.steps:
        evaluate_full_validation()


def cmd_infer(args: argparse.Namespace) -> None:
    import json

    from business_entity_resolution.config import MODELS_DIR
    from business_entity_resolution.pipeline.inference_pipeline import run_inference
    from business_entity_resolution.pipeline.ml_inference import run_ml_inference

    spec = json.loads((MODELS_DIR / args.model).read_text(encoding="utf-8"))
    if spec.get("type") == "ml":
        report = run_ml_inference(spec_file=args.model, validate=not args.no_validate)
    else:
        report = run_inference(model_file=args.model, validate=not args.no_validate)
    print(json.dumps({k: v for k, v in report.items() if k != "validator_output"}, indent=1, default=str))
    if "validator_output" in report:
        print(report["validator_output"])


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
    t = sub.add_parser("train", help="Stages 5-18: blocking analysis, features, XGBoost vs LightGBM, calibration, policy, ablation")
    t.add_argument("--steps", nargs="+", default=["build", "blocking", "model"], choices=["build", "blocking", "model", "fullval"])
    t.add_argument("--no-ablation", action="store_true")
    t.set_defaults(func=cmd_train)
    i = sub.add_parser("infer", help="full test inference -> output/matching_results.tsv + candidate_pairs.tsv")
    i.add_argument("--model", default="matcher_spec.json", help="spec in models/: matcher_spec.json (ML) or baseline_rule.json")
    i.add_argument("--no-validate", action="store_true", help="skip the official validator")
    i.set_defaults(func=cmd_infer)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
