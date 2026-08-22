# Study design

> **Partly superseded by P1 (D-043).** The research questions, conditions,
> counterfactual contract and ground-truth construction below remain binding. The
> primary outcome is no longer the 0–100 suitability score: it is the Yes/No token
> log-odds contrast (D-044), estimated on objectively qualified candidates (D-047),
> with employer-context richness crossed in and the interaction as the primary
> estimand (D-046). "Scope of the pilot" describes P0 and is closed. RQ3 is
> deferred (D-043).

## The question

Large language models shift their decisions when a candidate's name changes.
That much is established. What is not established is the relationship between
that shift and the model's own account of it: whether a model asked "did the
name affect your evaluation?" gives an answer that tracks the influence
independently measurable in its own behaviour.

The study measures three things on the same trials, in one decision domain
(United States entry-to-mid-level hiring) and on one identity dimension
(race and ethnicity conveyed by personal names, White-associated versus
Black-associated).

## Research questions

**RQ1 — Cue sensitivity.** Do favourable-decision rates and suitability scores
differ between matched counterfactual conditions that vary only the applicant's
name, or only a directly stated racial identity?

**RQ2 — Encoding.** Does the size of that difference depend on whether identity
is implied by a name or stated outright? A cue effect that disappears under
direct statement implies that trained fairness behaviour is keyed to explicit
mention and does not generalise to proxies. That is the deployment-relevant
failure mode, because production systems receive names, not demographic fields.

**RQ3 — Self-report validity.** Does a model's structured claim about having
been influenced discriminate cells where a cue measurably moved the decision
from cells where it did not? Reported as AUROC with a calibration curve.

**RQ4 — Revision quality.** Does naming identity in a reflection turn improve
accuracy against the qualification rule relative to a generic reconsideration
prompt, and at what cost in unnecessary revisions?

**RQ5 — Credential attenuation.** How much of the name effect survives an
explicit high-status credential signal?

## Conditions

Five identity conditions per scenario, crossed with two credential-prestige
levels. Conditions are presented one at a time in independent sessions; matched
variants are never shown side by side.

| Condition | Identity block |
|---|---|
| Neutral | `Applicant identifier: Candidate A` |
| Concealed White | `Applicant name: <White-associated name>` |
| Concealed Black | `Applicant name: <Black-associated name>` |
| Direct White | Neutral identifier plus a stated identity |
| Direct Black | Neutral identifier plus a stated identity |

Within a counterfactual set every other byte of the prompt is identical. This
is verified before inference, not asserted: see `02_SCENARIO_CONTRACT.md`.

A name cue is never combined with a direct identity statement.

## Reflection

Each trial is randomised to one of two second turns.

*Generic* asks the model to reconsider without mentioning identity, names,
fairness, or bias. Models revise on reflection whether or not identity is
raised, so a revision rate measured only under an identity-naming prompt
confounds bias correction with ordinary answer instability and with compliance
to a leading premise. The generic arm is what makes the identity arm
interpretable.

*Identity-specific* names the cue and asks for a structured assessment, a
confidence rating, and an optional revised decision and score.

Neutral-condition trials take a *late-disclosure* arm instead of the
identity-specific one. Asking whether an identity cue influenced a decision
taken without any identity cue has exactly one correct answer and measures
nothing. Supplying the identity afterwards measures reaction to genuinely new
information, which is a different quantity and is reported separately.

## Ground truth

Two constructs, both observable, neither annotated.

**The correct decision** is computed from the scenario's hard gates. No human
judges it.

**Whether a cue moved the model** is estimated from matched counterfactual
differences for that model, against its own repeated-run variation. No human
judges it either, and the model's stated reason is never used: an explanation is
a claim about a process, and testing that claim against behaviour is the point
of the study, so treating the explanation as evidence of the behaviour would
assume the answer.

## The primary outcome, and why

The 0-100 suitability score is the primary outcome. The binary decision is
secondary.

A decision flip is a lossy binarisation that discards every movement not
crossing the threshold, and published demographic flip rates are on the order of
a few percentage points. A binary-powered design at that effect size needs
observation counts roughly two orders of magnitude beyond what is feasible here.
A paired within-scenario comparison on a continuous score detects a standardised
effect near 0.4 with roughly fifty scenarios.

This only holds if the score has movement the qualification rule does not
already fix. That property is built into the scenario contract and verified by
the blocking diagnostics in `05_PILOT_GATES_AND_ANALYSIS.md`. It is not assumed.

## What this design does not claim

The concealed condition estimates the effect of a **name cue as deployed**. A
name carries perceived race together with perceived class, citizenship, and
associations specific to the string. Perceived class is downstream of perceived
race rather than a rival cause, so adjusting for it would remove part of the
effect being estimated. Class is therefore manipulated directly, through the
credential factor, and the estimand is stated as the bundled cue effect.

Nothing here is a claim about a model's internal states. The measured quantities
are sensitivity to specific textual encodings, the diagnostic value of a
structured self-report, and the quality of a revision.

## Scope of the pilot

The pilot exists to produce five numbers, not to test a hypothesis:

1. the standard deviation of the paired within-scenario score difference;
2. the near-threshold binary flip rate;
3. the score's distributional behaviour per model;
4. rule-determinacy of the score;
5. the score's response to a legitimate change in the soft criteria.

Every sizing decision for a confirmatory run depends on these. The pilot is
exploratory by declaration, and its results are used only for variance
estimates and sample sizing.
