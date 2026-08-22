# Limitations

Stated where they affect interpretation, not collected at the end as a
formality. Several of them bound what the results can be used to say.

## The name cue is a bundled treatment

A name carries perceived race together with perceived class, citizenship, and
associations specific to the string, and in United States names the correlation
between perceived race and perceived class is strong enough that no selection or
adjustment procedure separates them cleanly at usable sample sizes.

Statistical control is not merely insufficient here, it is inadvisable:
perceived race plausibly causes perceived class, which makes class perception a
mediator, and adjusting for a mediator introduces post-treatment bias. The same
defect applies to matching on those ratings.

The estimand is therefore the effect of a name cue as deployed, which is also
the quantity that matters for deployment, since production systems receive names
rather than decomposed attributes. The credential factor bounds how much of it
survives an explicit high-status signal. Concealed-condition estimates are not
the isolated causal effect of race and are not reported as such.

## Direct cues are not a clean comparison either

Stating an identity outright removes ambiguity about which group is intended,
but it also makes the fairness test conspicuous and can activate trained safety
behaviour. The concealed-versus-direct contrast therefore contains a disclosure
difference as well as a difference in how identity is encoded.

This is why guardrail activation is measured per condition rather than noted as
a caveat. Agreement across cue modes strengthens interpretation; disagreement is
reported rather than averaged away, and the guardrail rates are what allow the
interaction to be attributed.

## Nothing here observes an internal state

The measured quantities are sensitivity to specific textual encodings, the
diagnostic value of a structured self-report, and the quality of a revision. A
model's explanation is retained for qualitative error analysis and is never
treated as evidence of what produced its answer.

## The scenarios are synthetic and the source records have defects

Requirements are drawn from synthetic job records, some of which assemble
required experience from unrelated occupations. Four occupations are excluded on
that basis and the reasons are recorded per occupation. The retained six have
been read and are coherent, but they have not been reviewed by anyone with
hiring-domain expertise.

Objective qualification rules improve reproducibility and simplify real hiring,
where requirements are negotiable and evaluators hold private information.
Nothing in this study justifies automated hiring decisions in deployment.

## The domain cannot be casually expanded

The design assumes demographic information is normatively irrelevant to the
correct decision. That holds for hiring. It does not hold uniformly across the
domains originally considered: clinical guidelines are legitimately sex- and
sometimes ancestry-stratified for certain conditions, so medical triage has no
clean baseline where identity should change nothing, and cannot be added without
a different definition of ground truth. Any domain expansion must first
establish that the normative baseline holds.

## Discretion has a cost and it lands on the label

Widening the score's discretionary room is what gives a cue space to act, but it
also widens the base each counterfactual difference sits on. The per-cell
cue-sensitivity label that the self-report is scored against therefore gets
noisier, and label noise attenuates AUROC. Defining the positive class from a
shrunken per-cell estimate mitigates this; it does not eliminate it, and
reported AUROC should be read as a lower bound.

## Structural limits that no design choice removes

- Public prompts and released materials can enter later training data.
- Served model versions change; exact identifiers and access dates are recorded
  for that reason and re-measurement is expected.
- Two model families is enough to estimate variance components and not enough to
  characterise a population of models.
- Results are bounded by the selected cultural and legal context, identity
  dimension, occupation set, prompt wording, and exact model versions.

## Currently unmet

- The name pool must be built from the validated perception data. Runs against
  the provisional pool are flagged and the diagnostics refuse to report them.
- The prestige stimuli have not been pretested for perceived standing or
  screened for unintended coding. Until they are, RQ5 is exploratory.
- The guardrail keyword screen has not been validated against hand coding.
- Occupation-level generalisation rests on six occupations rather than the
  twelve the sizing floor was originally written around. The cost is
  generalisation and not sensitivity, and occupation-level variability is
  reported so a reader can judge it.
