# Concealed and direct identity cues in LLM hiring decisions

A counterfactual audit of how open-weight language models respond to identity
cues in hiring decisions.

Each scenario pairs a synthetic applicant with an explicit qualification rule
whose binary outcome is computed programmatically from the applicant's
task-relevant facts. Five matched prompt conditions vary only the applicant's
identity — absent, implied by a name, or stated directly — crossed with two
credential-prestige levels and two levels of employer-context richness. The
measured quantity is the Yes/No token log-odds contrast at a fixed answer
position, read from a single forward pass rather than sampled, and the primary
estimand is the interaction between context richness and the identity cue among
candidates the objective rule qualifies.

## Layout

    configs/    every value that affects what is measured
    prompts/    prompt templates, response schemas, prompt manifest
    data/       scenario source records, generated scenarios, stimuli
    src/        pipeline
    tests/      structural, instrument and estimator tests, no model required
    k8s/        cluster manifests
    scripts/    one-off data preparation

## Getting started

    pip install -e .
    pytest
    python -m hiringcue.cli validate

`validate` checks the scenario set and the name pool against their structural
contracts and needs no model or accelerator. Torch is not a declared dependency;
it is supplied by the accelerator image, and every stage except the readout
itself runs without it.

## Stages

    python -m hiringcue.cli plan                             render prompts, check integrity, write batches
    python -m hiringcue.cli stage0    --model <key>          answer boundary, variants, batch stability
    python -m hiringcue.cli measure   --model <key> --label <label>
    python -m hiringcue.cli diagnose  --readings <dir>       blocking diagnostics
    python -m hiringcue.cli calibrate                        null rejection rate across the registered grid
    python -m hiringcue.cli size                             variance-only confirmatory sizing

Each stage writes to disk and the next reads it.

## Before collecting measurements

`data/stimuli/name_pairs.json` must be built from the Validated Names perception
data with `scripts/build_name_stimuli.py`. The published data is not
redistributed here. The pool is a set of pairs matched across arms on
attribution accuracy; there is no substitute, because an unmatched pool
reintroduces the arm asymmetry the matching removes.
