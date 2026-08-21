"""Command-line entry points for each stage of the experiment.

    validate   structural checks on the scenario set and the name pool
    plan       render every prompt, check counterfactual integrity, write the plan
    stage0     tokenizer, answer boundary, and cross-batch stability per checkpoint
    measure    run a planned round through the readout and record it
    diagnose   evaluate the blocking diagnostics on collected readings
    calibrate  null rejection rate of the estimator at every registered grid cell
    size       apply the variance-only sizing rule to development variance estimates

Each stage writes its output to disk and the next stage reads it, so a round can
stop and resume at a stage boundary without redoing the stage before it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import (
    config,
    context,
    diagnostics,
    estimate,
    paths,
    plan,
    scenarios,
    stimuli,
    twins,
)


def _plan_dir(args) -> Path:
    return Path(args.plan_dir) if args.plan_dir else paths.OUTPUT / "plan"


def _load_readings(path: Path) -> list[dict]:
    if path.is_dir():
        records: list[dict] = []
        for file in sorted(path.glob("readings__*.jsonl")):
            records.extend(
                json.loads(line) for line in file.read_text().splitlines() if line
            )
        if not records:
            raise SystemExit(f"no readings__*.jsonl under {path}")
        return records
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def cmd_validate(args) -> int:
    families = scenarios.validated_families()
    excluded = scenarios.excluded_occupations()
    required = config.study()["qualification"]["primary_estimand_requires_gold"]

    by_band: dict[str, int] = {}
    for family in families:
        by_band[family.margin_band] = by_band.get(family.margin_band, 0) + 1
    qualified = [family for family in families if family.gold_decision == required]

    pairs = stimuli.load_pairs()
    development = stimuli.load_pairs(stimuli.DEVELOPMENT)
    confirmatory = stimuli.load_pairs(stimuli.CONFIRMATORY)
    stimuli.disjoint(development, confirmatory)

    print(f"scenario families:   {len(families)}")
    print(f"  qualified (primary): {len(qualified)}")
    print(f"  rule control:        {len(families) - len(qualified)}")
    print(f"occupations:         {len({f.occupation_slug for f in families})}")
    print(f"margin bands:        {by_band}")
    print(f"name pairs:          {len(pairs)}")
    print(f"  development:         {len(development)}")
    print(f"  confirmatory:        {len(confirmatory)}")
    print(f"context levels:      {list(context.levels())}")
    print(f"realistic variants:  {list(context.realistic_variants())}")
    if excluded:
        print(f"excluded:            {len(excluded)}")
        for slug, reason in excluded.items():
            print(f"  - {slug}: {' '.join(reason.split())[:110]}")
    return 0


def cmd_plan(args) -> int:
    families = scenarios.validated_families()
    if args.occupations:
        requested = [slug.strip() for slug in args.occupations.split(",") if slug.strip()]
        available = {family.occupation_slug for family in families}
        unknown = sorted(set(requested) - available)
        if unknown:
            raise SystemExit(
                f"unknown or excluded occupation(s): {unknown}; available: {sorted(available)}"
            )
        families = [family for family in families if family.occupation_slug in requested]

    twin_profiles = twins.load()
    if not twin_profiles:
        print("no soft twins found; the free-parameter diagnostic will be unavailable")
    pairs = stimuli.load_pairs(args.pool)
    # The development-only context levels are a stimulus probe, not part of the
    # estimand. Pairing them with the confirmatory name pool would put them in a
    # frozen collection, so the combination is refused rather than warned about.
    if args.development_contexts and args.pool == stimuli.CONFIRMATORY:
        raise SystemExit(
            "--development-contexts is not available on the confirmatory pool; the "
            "development-only context levels never enter the confirmatory estimand"
        )
    prompts = plan.build(
        families=families,
        pairs=pairs,
        twins=twin_profiles,
        realistic_variant=args.realistic_variant,
        development_contexts=args.development_contexts,
    )
    slots = plan.batch_manifest(prompts)

    out = _plan_dir(args)
    plan.write(prompts, slots, out)
    summary = plan.summarise(prompts, slots)
    summary["name_pool"] = stimuli.pool_summary(pairs)
    summary["pool_role"] = args.pool
    (out / "plan_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {len(prompts)} prompts in {summary['batches']} batches to {out}")
    print(json.dumps({key: summary[key] for key in ("by_role", "name_pairs", "context_variant")}, indent=2))
    return 0


def cmd_stage0(args) -> int:
    from . import stage0

    path = stage0.run(
        model_key=args.model,
        plan_dir=_plan_dir(args),
        out_dir=Path(args.out) if args.out else paths.OUTPUT / "stage0",
        model_path=args.model_path,
        tokenizer_only=args.tokenizer_only,
    )
    report = json.loads(path.read_text())
    print(json.dumps({key: value for key, value in report.items() if key != "readings"}, indent=2))
    return 0 if report["verdict"] == diagnostics.PASS else 1


def cmd_measure(args) -> int:
    from . import measure

    path = measure.run(
        model_key=args.model,
        plan_dir=_plan_dir(args),
        out_dir=Path(args.out) if args.out else paths.OUTPUT / "readings",
        label=args.label,
        pool_role=args.pool,
        model_path=args.model_path,
        limit=args.limit,
    )
    print(f"wrote {path}")
    print(json.dumps(measure.agreement(path), indent=2))
    return 0


def cmd_diagnose(args) -> int:
    from . import measure

    readings = Path(args.readings)
    rows = _load_readings(readings)
    out_dir = Path(args.out) if args.out else paths.OUTPUT / "analysis"
    measured = {}
    for file in sorted(readings.glob("readings__*.jsonl")) if readings.is_dir() else [readings]:
        model = json.loads(file.read_text().splitlines()[0])["model_key"]
        measured[model] = measure.agreement(file)
    report = diagnostics.evaluate(rows, agreement=measured)
    for model, entry in report["models"].items():
        model_rows = [row for row in rows if row["model_key"] == model]
        entry["development"] = diagnostics.development_estimates(model_rows)
    if args.stage0:
        stage0_report = json.loads(Path(args.stage0).read_text())
        disclosed = stage0_report.get("batch_size_sensitivity")
        if disclosed is not None:
            report["models"][stage0_report["model_key"]][
                "batch_size_sensitivity"
            ] = disclosed
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    blocked = [
        model for model, entry in report["models"].items() if not entry["authorised"]
    ]
    if blocked:
        print(f"\nnot authorised to proceed: {blocked}")
        return 1
    return 0


def _components(args) -> estimate.VarianceComponents:
    if args.readings:
        cells = estimate.interaction_cells(_load_readings(Path(args.readings)))
        values, _, _ = estimate.cell_matrix(cells)
        return estimate.components(values)
    return estimate.VarianceComponents(
        family=args.family_sd**2, name=args.name_sd**2, residual=args.residual_sd**2
    )


def cmd_calibrate(args) -> int:
    settings = config.study()["sizing"]
    low, high = (float(value) for value in settings["calibration_interval"])
    out_dir = Path(args.out) if args.out else paths.OUTPUT / "analysis"

    report = {"interval": [low, high], "scenarios": {}}
    failures = []
    for scenario in settings["calibration_scenarios"]:
        parts = estimate.VarianceComponents(
            family=float(scenario["family_sd"]) ** 2,
            name=float(scenario["name_sd"]) ** 2,
            residual=float(scenario["residual_sd"]) ** 2,
        )
        rates = estimate.calibration(parts)
        report["scenarios"][scenario["label"]] = {
            f"F{families}xJ{names}": rate for (families, names), rate in sorted(rates.items())
        }
        failures.extend(
            f"{scenario['label']} F{families}xJ{names}={rate:.4f}"
            for (families, names), rate in sorted(rates.items())
            if not low <= rate <= high
        )

    report["failures"] = failures
    report["accepted"] = not failures
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sizing_calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


def cmd_size(args) -> int:
    settings = config.study()["sizing"]
    out_dir = Path(args.out) if args.out else paths.OUTPUT / "analysis"
    grid = (
        [int(value) for value in settings["name_grid"]]
        if args.registered_grid
        else [int(value) for value in settings["available_name_grid"]]
    )
    report = estimate.select_design(_components(args), name_grid=grid)
    report["name_grid_source"] = "registered" if args.registered_grid else "available"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "confirmatory_sizing.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["decision"] == "collect" else 1


def cmd_submit(args) -> int:
    """Render a job manifest for one checkpoint, with its own accelerator constraint.

    The manifests carry placeholders rather than a checkpoint's identifiers, and
    substituting them by hand is how a stale revision or the wrong accelerator
    constraint reaches a run. The device products come from the checkpoint's own
    entry, so a checkpoint that fits a 40GB device is not queued behind the 80GB
    pool because a manifest written for a 24B model said it had to be.
    """
    from . import stage0

    entry = stage0.model_entry(args.model)
    defaults = config.models()["defaults"]
    products = entry.get("gpu_products", defaults["gpu_products"])

    manifest = (paths.ROOT / "k8s" / f"job-{args.stage}.yaml").read_text()
    substitutions = {
        "__MODEL_KEY__": args.model,
        "__MODEL_ID__": entry["model_id"],
        "__MODEL_REVISION__": entry["revision"],
        "__GPU_PRODUCTS__": "[" + ", ".join(products) + "]",
        "__CONTEXT_VARIANT__": args.context_variant or "",
        "__LABEL__": args.label or "",
    }
    for key, value in substitutions.items():
        manifest = manifest.replace(key, value)
    leftover = [key for key in substitutions if key in manifest]
    if leftover:
        raise SystemExit(f"manifest still carries {leftover}")
    print(manifest)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hiringcue", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate").set_defaults(func=cmd_validate)

    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--stage", required=True, choices=["stage0", "stage1"])
    p_submit.add_argument("--model", required=True)
    p_submit.add_argument("--label")
    p_submit.add_argument("--context-variant")
    p_submit.set_defaults(func=cmd_submit)

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--plan-dir")
    p_plan.add_argument(
        "--pool",
        choices=[stimuli.DEVELOPMENT, stimuli.CONFIRMATORY],
        default=stimuli.DEVELOPMENT,
        help="which disjoint name pool this round draws from",
    )
    p_plan.add_argument(
        "--realistic-variant",
        help="context template bound to the realistic level; defaults to the first "
        "in the predeclared order",
    )
    p_plan.add_argument(
        "--development-contexts",
        action="store_true",
        help="add the development-only context levels that measure whether the "
        "amplification depends on the employer being recognisable",
    )
    p_plan.add_argument(
        "--occupations",
        help="comma-separated retained occupation slugs for a bounded plan",
    )
    p_plan.set_defaults(func=cmd_plan)

    p_stage0 = sub.add_parser("stage0")
    p_stage0.add_argument("--model", required=True)
    p_stage0.add_argument("--plan-dir")
    p_stage0.add_argument("--out")
    p_stage0.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="run the boundary and variant enumeration without loading weights",
    )
    p_stage0.add_argument(
        "--model-path",
        help="local snapshot path; overrides the pinned revision lookup",
    )
    p_stage0.set_defaults(func=cmd_stage0)

    p_measure = sub.add_parser("measure")
    p_measure.add_argument("--model", required=True)
    p_measure.add_argument("--label", required=True)
    p_measure.add_argument("--plan-dir")
    p_measure.add_argument("--out")
    p_measure.add_argument("--limit", type=int)
    p_measure.add_argument(
        "--pool",
        choices=[stimuli.DEVELOPMENT, stimuli.CONFIRMATORY],
        default=stimuli.DEVELOPMENT,
    )
    p_measure.add_argument(
        "--model-path",
        help="local snapshot path; overrides the pinned revision lookup",
    )
    p_measure.set_defaults(func=cmd_measure)

    p_diagnose = sub.add_parser("diagnose")
    p_diagnose.add_argument("--readings", required=True)
    p_diagnose.add_argument("--out")
    p_diagnose.add_argument(
        "--stage0",
        help="Stage 0 report whose mandatory batch-size sensitivity is copied beside this result",
    )
    p_diagnose.set_defaults(func=cmd_diagnose)

    p_calibrate = sub.add_parser("calibrate")
    p_calibrate.add_argument("--out")
    p_calibrate.set_defaults(func=cmd_calibrate)

    p_size = sub.add_parser("size")
    p_size.add_argument(
        "--readings", help="development readings; variance components are taken from them"
    )
    p_size.add_argument("--family-sd", type=float, default=0.45)
    p_size.add_argument("--name-sd", type=float, default=0.25)
    p_size.add_argument("--residual-sd", type=float, default=0.50)
    p_size.add_argument(
        "--registered-grid",
        action="store_true",
        help="size against the full registered name grid rather than the counts the "
        "matched pool can supply",
    )
    p_size.add_argument("--out")
    p_size.set_defaults(func=cmd_size)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
