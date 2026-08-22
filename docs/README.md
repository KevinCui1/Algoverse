# Documentation

Binding specification for the study. Read in this order before changing
anything that affects what is measured.

| File | Covers |
|---|---|
| `01_STUDY_DESIGN.md` | Research questions, estimands, conditions, what is and is not claimed |
| `02_SCENARIO_CONTRACT.md` | The two-layer qualification rule, the emit-time contract, source data defects |
| `03_STIMULI_AND_PROMPTS.md` | Name and prestige stimuli, prompt pack, response schemas, parsing |
| `04_EXECUTION.md` | Models, sampling, batching, cluster jobs, run manifests |
| `05_PILOT_GATES_AND_ANALYSIS.md` | The blocking diagnostics, derived fields, reported metrics |
| `06_LIMITATIONS.md` | What the design cannot rule out |
| `07_DECISIONS.md` | Append-only record of decisions and their reasoning |

`07_DECISIONS.md` takes precedence over the other files where they conflict:
it is where later corrections are recorded, and the earlier text is left in
place rather than silently rewritten.

**This precedence rule is currently load-bearing.** D-043 reopens the study as P1
and D-044 onward change the primary outcome, the measurement path, the candidate
population, the stimulus factors, the inference, the target and the model set.
Files 01, 04 and 05 carry banners marking what they no longer govern, but the
binding text is in `07_DECISIONS.md`. Read D-043 through the last entry before
acting on anything in 01 through 06.

The implementation of the superseded path has been deleted rather than left in
place (D-058), so following those files in code raises an import error rather
than returning a number on a retired scale. The prose is what still has to be
read with the banners in mind.
