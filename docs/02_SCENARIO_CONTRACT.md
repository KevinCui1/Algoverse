# Scenario contract

## The problem this solves

Two properties are easy to conflate and only one of them is wanted.

*The gold decision must be computable by the analyst.* This is what removes
human annotation of causation from the design.

*The task must not be mechanically solvable from the prompt.* If the prompt
contains visible arithmetic whose answer is forced, an identity cue has no free
parameter to act on. A cue that cannot move anything cannot be measured, and the
study's primary outcome carries no signal.

These do not conflict once the analyst's information set is separated from the
model's. The generator can hold a weighting that is never shown to the model:
the gold decision is then fully programmatic while the task as presented still
requires weighing criteria with no supplied weights.

## The two-layer rule

**Hard gates** determine the binary decision. Two to four per scenario, each
categorical or a threshold on a named countable, individually checkable from the
prompt, carrying no weighting question. The gold decision is the conjunction of
the gates and reads nothing else — not the soft criteria, not credential
prestige, not any identity field.

**Soft criteria** determine nothing. Three to five non-commensurable evaluative
dimensions on which the candidate is deliberately mixed. The prompt states them
and states that they bear on the score. It supplies no weights, no ranking, and
no ordering that implies one.

The suitability score is therefore not determined by the qualification rule. The
rule fixes advance or do-not-advance; nothing in it dictates whether a
gate-passing candidate is a 61 or an 84.

## Ambiguity is structural, not stochastic

Where the discretion sits decides what it costs.

*Stochastic* discretion — a different score on a repeated identical prompt —
inflates the paired within-scenario difference directly, which is the quantity
the whole sample-size calculation rests on. It is expensive and it is not the
target.

*Structural* discretion — genuinely non-commensurable criteria with no supplied
weights, so a reasonable evaluator could land anywhere in a wide band — inflates
between-scenario variance while leaving the paired difference alone, because the
ambiguity is identical in both members of a counterfactual pair and cancels in
the subtraction.

The design rule follows: **ambiguity must be constant within a counterfactual
pair and variable across pairs.** The soft-criteria block is non-identity
content and is held byte-identical across the conditions of a scenario by the
integrity check below, so this holds by construction.

## Emit-time contract

Enforced in `src/hiringcue/scenarios.py` and run before any prompt is rendered.

| | Requirement |
|---|---|
| S1 | The gold decision is a pure function of the hard gates. Permuting the soft layer returns an identical result. |
| S2 | The soft profile is mixed: at least one criterion above expectation and one below. A uniform profile is scalarisable and reintroduces visible arithmetic. |
| S3 | Soft criteria span at least three distinct occupational dimension families, so no common unit permits mechanical aggregation. |
| S4 | Every hard gate declares a minimum margin unit. |
| S5 | Near-threshold candidates sit no more than one minimum unit from the bar. |
| S6 | No identifier is shared between the gate set and the soft criteria. |
| S7 | Near-threshold families are at or above the set median ambiguity score. Near the bar the arithmetic is most conspicuous, so a scenario that is both near-threshold and thin on soft criteria is the worst case in the set and is not emitted. |

## Counterfactual integrity

Enforced in `src/hiringcue/render.py` before inference, because after inference
the compute is spent.

Every prompt is hashed twice: once as rendered, and once with the identity block
replaced by a fixed token. Within a counterfactual set — the five conditions of
one family at one prestige level — the normalised hashes must all match and the
exact hashes must all differ. A normalised mismatch means something other than
identity varies between conditions, and the comparison is not a counterfactual.

Three further screens run on the same pass. Neutral prompts must contain no
name, no race term, and no demographic placeholder, because the baseline every
cue effect is measured against would otherwise be contaminated. Concealed
prompts must carry no direct demographic descriptor. Direct prompts must use the
neutral identifier rather than a race-associated name.

## Gate wording

Some source records phrase a numeric requirement without stating its number
("Experience as a Chef or Head Cook" for a two-year minimum). A reader of that
prompt cannot tell whether the applicant's reported value clears the bar, which
destroys the property the design rests on — that the correct decision is
determinable from the prompt without judgement.

An undeterminable gate is also precisely where a cue can move a decision while
looking like bias, so leaving it in place would inflate the measured effect.
The threshold is therefore restored at render time from the structured value
whenever the wording omits it, and a test asserts that every numeric gate's
rendered text contains its threshold.

## Known defects in the source records

The job records are synthetic. For some occupations the required-experience list
is assembled from unrelated occupation titles, producing requirements no
competent evaluator would accept: a pediatric surgeon required to have four
years as a materials scientist, an aerospace engineer required to have worked as
a food-roasting machine operator, an airline pilot required to have captained a
ship.

This matters for two reasons beyond appearance. The gold decision is only
objective if a competent evaluator would agree the rule is the rule; when a
requirement is incoherent, a model that overrides it is being reasonable rather
than wrong, and initial accuracy stops measuring accuracy. And an incoherent
requirement makes the task ambiguous in a way that has nothing to do with
identity, widening the space a cue appears to act in.

Four occupations are excluded on this basis, with the reason recorded per
occupation in `configs/scenario_exclusions.yaml`. Individual gates are not
dropped from an otherwise sound occupation: removing a gate changes which gate
is decisive and invalidates the margin band the candidate was generated to
occupy.

Six occupations are retained, giving twenty-four scenario families across all
four margin bands — within the pilot's target range, so the pilot can produce
its variance estimates without regeneration.

Regeneration for a confirmatory set must source requirements from the
occupation's own task, skill, and knowledge profile rather than from the source
record's experience list, and must reject any requirement naming an occupation
other than the target.

## Perturbed twins

The twin is the positive control for the score's free parameter: the same
occupation, the same applicant, the same hard gates, with the non-binding
evaluative evidence changed. A model that will not move its score for a
substantive change to the criteria the score is supposed to reflect will not
move it for a name either, and no sample size repairs that, because the problem
is in the numerator rather than the denominator.

Twins are authored rather than templated. A template-written sentence sitting
among generated prose differs in register as well as substance, and a score
change could then be attributed to the writing. The authoring pass may not touch
anything countable — years, degrees, certifications, licences, job titles all
belong to the gate layer — and a twin mentioning one is rejected.

The soft profile varies across margin bands within an occupation, so a twin is
authored per family rather than per occupation. The stored pool is refused at
load time when it is not the perturbation of the scenario set it is loaded with,
so a pool authored against superseded profiles cannot be used silently.
