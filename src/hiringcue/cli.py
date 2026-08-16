"""Command-line entry points for each stage of the experiment.

    validate   structural checks on the scenario set, no model required
    plan       render every prompt, check counterfactual integrity, write the plan
    twins      author the perturbed soft-criteria twins
    run        execute both turns for one model
    analyse    parse responses and write derived fields
    gates      evaluate the blocking pilot diagnostics

Each stage writes its output to disk and the next stage reads it, so a run can
stop and resume at a stage boundary without redoing the stage before it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config, derive, diagnostics, paths, pilot, plan, scenarios, stimuli, twins


def _plan_dir(args) -> Path:
    return Path(args.plan_dir) if args.plan_dir else paths.OUTPUT / "plan"


def cmd_validate(args) -> int:
    families = scenarios.validated_families()
    excluded = scenarios.excluded_occupations()
    by_band: dict[str, int] = {}
    for family in families:
        by_band[family.margin_band] = by_band.get(family.margin_band, 0) + 1
    print(f"scenario families: {len(families)}")
    print(f"occupations:       {len({f.occupation_slug for f in families})}")
    print(f"margin bands:      {by_band}")
    if excluded:
        print(f"excluded:          {len(excluded)}")
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
    variants, trials = plan.build(
        families=families, twins=twin_profiles, allow_provisional=args.allow_provisional
    )
    out = _plan_dir(args)
    plan.write(variants, trials, out)

    pool_path = stimuli.name_pool_path(allow_provisional=args.allow_provisional)
    pool = stimuli.load_names(allow_provisional=args.allow_provisional)
    summary = {
        "families": len(families),
        "occupations": sorted({family.occupation_slug for family in families}),
        "variants": len(variants),
        "trials": len(trials),
        "name_pool": pool_path.name,
        "name_pool_provisional": stimuli.is_provisional(pool_path),
        "name_pool_summary": stimuli.pool_summary(pool),
        "seed": config.study()["seed"],
        "runs_per_variant": config.study()["runs_per_variant"],
    }
    (out / "plan_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {len(variants)} variants and {len(trials)} trials to {out}")
    if summary["name_pool_provisional"]:
        print("WARNING: provisional name pool; diagnostics will refuse to report")
    return 0


def cmd_twins(args) -> int:
    from .twin_author import author_all

    families = scenarios.validated_families()
    profiles = author_all(families, model_key=args.model)
    path = twins.save(profiles)
    print(f"wrote {len(profiles)} twin profiles to {path}")
    return 0


def cmd_run(args) -> int:
    from . import runner

    path = runner.run(
        model_key=args.model,
        plan_dir=_plan_dir(args),
        out_dir=Path(args.out) if args.out else paths.OUTPUT / "responses",
        run_label=args.label,
        limit=args.limit,
        model_path=args.model_path,
    )
    print(f"wrote {path}")
    return 0


def _collect(response_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(response_dir.glob("responses__*.jsonl")):
        rows.extend(derive.load_responses(path))
    if not rows:
        raise SystemExit(f"no response files under {response_dir}")
    return rows


def _manifests(response_dir: Path) -> list[dict]:
    manifests = [
        json.loads(path.read_text())
        for path in sorted(response_dir.glob("manifest__*.json"))
    ]
    if not manifests:
        raise SystemExit(f"no run manifests under {response_dir}")
    return manifests


def cmd_analyse(args) -> int:
    response_dir = Path(args.responses) if args.responses else paths.OUTPUT / "responses"
    out_dir = Path(args.out) if args.out else paths.OUTPUT / "analysis"
    rows = derive.parse_records(_collect(response_dir))
    derive.write(rows, out_dir / "trials_parsed.jsonl")

    differences = derive.counterfactual_differences(rows)
    derive.write(differences, out_dir / "counterfactual_differences.jsonl")

    invalid = sum(1 for row in rows if not row.get("initial_valid"))
    summary = {
        "trials": len(rows),
        "schema_failures_initial": invalid,
        "schema_failures_reflection": sum(
            1 for row in rows if not row.get("reflection_valid")
        ),
        "initial_accuracy": {
            model: round(
                sum(
                    1
                    for row in rows
                    if row["model_key"] == model
                    and row.get("soft_variant") != "twin"
                    and row.get("initial_valid")
                    and row.get("initial_correct")
                )
                / max(
                    sum(
                        1
                        for row in rows
                        if row["model_key"] == model
                        and row.get("soft_variant") != "twin"
                        and row.get("initial_valid")
                    ),
                    1,
                ),
                4,
            )
            for model in sorted({row["model_key"] for row in rows})
        },
        "paired_score_difference_sd_concealed": derive.paired_difference_sd(
            differences, "score_shift_concealed"
        ),
        "paired_score_difference_sd_direct": derive.paired_difference_sd(
            differences, "score_shift_direct"
        ),
        "cue_mode_interaction": derive.paired_difference_sd(
            differences, "cue_mode_interaction"
        ),
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pilot_report = pilot.summary(rows, differences, _manifests(response_dir))
    (out_dir / "pilot_summary.json").write_text(
        json.dumps(pilot_report, indent=2) + "\n"
    )
    sizing = pilot.confirmatory_sizing(differences)
    (out_dir / "confirmatory_sizing.json").write_text(
        json.dumps(sizing, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


def cmd_gates(args) -> int:
    response_dir = Path(args.responses) if args.responses else paths.OUTPUT / "responses"
    out_dir = Path(args.out) if args.out else paths.OUTPUT / "analysis"
    rows = derive.parse_records(_collect(response_dir))

    provisional = any(
        json.loads(path.read_text()).get("name_pool_provisional")
        for path in response_dir.glob("manifest__*.json")
    )
    if provisional and not args.allow_provisional:
        raise SystemExit(
            "at least one run used the provisional name pool. Diagnostics are refused "
            "because the pool carries no perception statistics and no selection "
            "threshold was applied. Build data/stimuli/names.json and re-run, or pass "
            "--allow-provisional to inspect the numbers without treating them as results."
        )

    temperature = float(config.models()["sampling"]["temperature"])
    report = diagnostics.evaluate(rows, temperature=temperature)
    authorised_models = {
        model
        for model, entry in report["models"].items()
        if entry["confirmatory_run_authorised"]
    }
    report["confirmatory_sizing"] = pilot.confirmatory_sizing(
        derive.counterfactual_differences(rows),
        authorised_models=authorised_models,
    )
    report["name_pool_provisional"] = provisional
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pilot_gates.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    blocked = [
        model
        for model, entry in report["models"].items()
        if not entry["confirmatory_run_authorised"]
    ]
    if blocked:
        print(f"\nblocked for confirmatory run: {blocked}")
        return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hiringcue", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate").set_defaults(func=cmd_validate)

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--plan-dir")
    p_plan.add_argument("--allow-provisional", action="store_true")
    p_plan.add_argument(
        "--occupations",
        help="comma-separated retained occupation slugs for a bounded dry-run plan",
    )
    p_plan.set_defaults(func=cmd_plan)

    p_twins = sub.add_parser("twins")
    p_twins.add_argument("--model", required=True)
    p_twins.set_defaults(func=cmd_twins)

    p_run = sub.add_parser("run")
    p_run.add_argument("--model", required=True)
    p_run.add_argument("--label", required=True)
    p_run.add_argument("--plan-dir")
    p_run.add_argument("--out")
    p_run.add_argument("--limit", type=int)
    p_run.add_argument(
        "--model-path",
        help="local snapshot path; its directory name must equal the configured revision",
    )
    p_run.set_defaults(func=cmd_run)

    p_analyse = sub.add_parser("analyse")
    p_analyse.add_argument("--responses")
    p_analyse.add_argument("--out")
    p_analyse.set_defaults(func=cmd_analyse)

    p_gates = sub.add_parser("gates")
    p_gates.add_argument("--responses")
    p_gates.add_argument("--out")
    p_gates.add_argument("--allow-provisional", action="store_true")
    p_gates.set_defaults(func=cmd_gates)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
