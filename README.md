# Concealed and direct identity cues in LLM hiring decisions

A counterfactual audit of how open-weight language models respond to identity
cues in hiring decisions, and of whether a model's own report of having been
influenced tracks the influence measurable in its behaviour.

Each scenario pairs a synthetic applicant with an explicit qualification rule
whose binary outcome is computed programmatically from the applicant's
task-relevant facts. Five matched prompt conditions vary only the applicant's
identity — absent, implied by a name, or stated directly — crossed with two
credential-prestige levels. A second turn asks the model to reconsider, either
without mentioning identity or by naming the cue, so that bias correction can be
separated from ordinary answer instability.

## Layout

    configs/    every value that affects what is measured
    prompts/    prompt templates, response schemas, prompt manifest
    data/       scenario source records, generated scenarios, stimuli
    src/        pipeline
    tests/      structural and diagnostic tests, no model required
    k8s/        cluster manifests for batched inference
    scripts/    one-off data preparation

## Getting started

    pip install -e .
    pytest
    python -m hiringcue.cli validate

`validate` checks the scenario set against its structural contract and needs no
model or accelerator.

## Stages

    python -m hiringcue.cli plan                        render prompts, check integrity
    python -m hiringcue.cli twins   --model <key>       author perturbed-twin controls
    python -m hiringcue.cli run     --model <key> --label <label>
    python -m hiringcue.cli analyse                     parse and derive
    python -m hiringcue.cli gates                       blocking pilot diagnostics

Each stage writes to disk and the next reads it.

## Before collecting measurements

`data/stimuli/names.json` must be built from the Validated Names perception data
with `scripts/build_name_stimuli.py`. A provisional pool ships for exercising the
pipeline; it carries no perception statistics, runs against it are flagged in the
run manifest, and the diagnostics refuse to report from it.
