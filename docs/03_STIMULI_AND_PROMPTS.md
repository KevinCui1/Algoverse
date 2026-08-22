# Stimuli and prompts

## Name cues

First-and-last-name stimuli from the Validated Names replication data, retained
only above a perception threshold recorded in `configs/stimuli.yaml` (0.75
correct group attribution; see D-010 for why it is not the 0.80 originally
declared), with at least twelve names per group. Below twelve, a single stimulus
can carry the result and the leave-one-name-out check stops being informative.

Attribution accuracy is not symmetric across the two groups, so a single
threshold retains 88 of 100 White-associated names and 22 of 100
Black-associated ones. The Black-associated pool is therefore the more selected
of the two; the arms are not matched on how typical their names are of their
group's corpus.

Names are assigned round-robin from a shuffled pool under the study seed, so
each stimulus appears a near-equal number of times across occupations, margin
bands, and prestige levels.

The published perception ratings for income, education, and citizenship are
carried through the pipeline and reported per group as a descriptive table.
They are **not** used to match names and **not** used as regression controls.
Perceived race plausibly causes perceived class, which makes those ratings
mediators rather than confounds; conditioning on them removes part of the effect
being estimated. Attribution is addressed by manipulation instead — see the
credential factor below.

### Acquiring the pool

The source data is not redistributed in this repository. Two files are needed
from the deposit, both published as CSV alongside their R serialisations:
`study123`, one row per respondent-name judgement pooled over the three surveys,
and `names`, the roster carrying each name's intended group.

    python scripts/build_name_stimuli.py --pooled study123.csv --names names.csv

The script writes `data/stimuli/names.json`. It stops rather than shipping a
defective pool in three cases: either group falling short of the configured
minimum, a declared perception attribute with no column in the source, and a
perception column that resolves but yields no numeric value — the last because
the deposit stores each perception both as answer text and as an ordinal
encoding, and reading the text column would produce a pool that looks complete
and reports an empty perception table.

A provisional pool (`data/stimuli/names.provisional.json`) exists so the
pipeline can be exercised end to end before the real data is in place. It
carries no measured perception statistics and no threshold has actually been
applied to it. Runs using it are flagged in the run manifest, and the pilot
diagnostics refuse to report against it unless explicitly overridden.

## Credential prestige

Two fictitious institutions, matched on every fact the qualification rule reads
and differing only in institutional standing. Neither is religiously,
regionally, or culturally distinctive, and the modest condition is not a
historically Black or otherwise culturally marked institution.

Prestige is a scenario-level attribute held constant across the counterfactual
name versions of a scenario, so it adds no prompt conditions and no additional
call volume. It serves three purposes: it implements the identification strategy
the bundled-treatment literature recommends, manipulating class rather than
adjusting for it; it supplies RQ5; and it provides a within-study reference
effect, since credential-prestige effects in model decisions are reported to be
substantially larger than demographic ones and are likely to be detectable even
in a run where the name effect is not.

Both stimuli should be pretested for perceived prestige and screened for
unintended racial, regional, or religious coding before a confirmatory run. Until
that pretest exists, treat RQ5 as exploratory.

## Prompt pack

`prompts/` holds every prompt as a text file with `{{PLACEHOLDER}}` fields.
`prompts/manifest.json` records which files pair with which, what placeholders
each requires, which response schema validates its output, and which repair
prompt applies. Rendering fails if any placeholder survives.

| Stage | Files |
|---|---|
| Scenario authoring | `scenario_author_system_v1.txt`, `scenario_author_user_v1.txt` |
| Soft-twin authoring | `soft_twin_author_system_v1.txt`, `soft_twin_author_user_v1.txt` |
| Initial decision | `initial_evaluation_system_v1.txt`, `initial_evaluation_user_v1.txt` |
| Reflection | `reflection_generic_v1.txt`, `reflection_identity_concealed_v1.txt`, `reflection_identity_direct_v1.txt`, `reflection_late_disclosure_v1.txt` |
| Repair | `repair_initial_json_v1.txt`, `repair_reflection_generic_json_v1.txt`, `repair_reflection_identity_json_v1.txt` |

The initial system prompt deliberately mentions neither identity, fairness,
protected classes, nor bias. Introducing any of them before the first decision
would prime the behaviour the study is trying to observe.

Soft-criterion presentation order is randomised once per scenario family under
the study seed and then held fixed across every condition and prestige level of
that family. Randomising per family stops a stable order from implying a
ranking; holding it fixed inside the family keeps the evaluative ambiguity
identical within each counterfactual pair.

## Response schemas and parsing

Each stage has a JSON Schema in `prompts/*.schema.json`. Responses are decoded
under the schema as a decoding constraint, so a malformed structure should not
occur; the schema check after the fact is a second line rather than the first.

That matters more than convenience. A parse failure drops a cell, and dropped
cells are not random with respect to condition: a direct identity statement is
the condition most likely to draw a refusal or a hedge, which is exactly where a
missing cell would distort the comparison. The repair prompts remain available
for a run where constrained decoding is unavailable, and are used at most once
per response, never for a substantive refusal, and never to request a different
decision.

Raw text is preserved for every response. Parsed fields are added alongside it,
never over it.

## Guardrail flags

Four behaviours are recorded per response and treated as outcomes rather than as
grounds for exclusion: refusal or partial refusal, unprompted fairness
commentary, explicit mention of the demographic attribute, and hedging.

They exist because safety activation is the leading alternative explanation for
the concealed-versus-direct interaction, which is the study's headline quantity.
If unprompted fairness commentary rises sharply in the direct conditions, the
interaction is at least partly a disclosure artefact, and the flags are what
allow that to be reported rather than guessed.

The screen is a keyword pass and is coarse by design. Its agreement with hand
coding must be measured on a stratified sample before its output appears in any
reported figure.
