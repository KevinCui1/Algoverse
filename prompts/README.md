# Hiring Evaluation Prompt Pack

This folder contains **all of the prompts needed for the project**. The prompts themselves are the `.txt` files.

Each text file is a complete prompt template. Reusable values are written as placeholders such as `{{APPLICANT_NAME}}` or `{{JOB_SUMMARY}}`. This avoids storing hundreds of nearly identical prompts when only a name, identity statement, job, or candidate record changes.

The JSON files do not replace or generate the prompt wording. They are optional tools that describe the templates and help software validate the model's responses.

## What the files do

### Scenario creation

- `scenario_author_system_v1.txt`
- `scenario_author_user_v1.txt`

Use these together to turn source job information into four candidate scenarios: clear fail, near fail, near pass, and clear pass.

### Initial hiring evaluation

- `initial_evaluation_system_v1.txt`
- `initial_evaluation_user_v1.txt`

Use these together to ask a model for its initial hiring decision, suitability score, confidence, and justification.

### Follow-up reflection

Choose exactly one after the initial evaluation:

- `reflection_generic_v1.txt` — asks the model to reconsider without mentioning identity.
- `reflection_identity_concealed_v1.txt` — asks whether a race-associated name influenced the evaluation.
- `reflection_identity_direct_v1.txt` — asks whether directly stated identity information influenced the evaluation.
- `reflection_late_disclosure_v1.txt` — introduces identity only after a neutral evaluation; use this as a separate control.

### JSON repair

- `repair_initial_json_v1.txt`
- `repair_reflection_generic_json_v1.txt`
- `repair_reflection_identity_json_v1.txt`

Use the matching repair prompt once if the model gives the right kind of answer in an invalid JSON format. Do not use a repair prompt for a refusal or to request a different decision.

## Simplest way to use the prompts

No programming is required.

1. Open the needed `.txt` file.
2. Copy its full contents.
3. Replace every `{{PLACEHOLDER}}` with the appropriate value.
4. Send the system prompt as the system message and the user prompt as the user message.
5. Save the model's raw JSON response.
6. For an evaluation trial, send one reflection prompt as the next user message in the same conversation.

For example:

```text
Applicant name: {{APPLICANT_NAME}}
```

could become:

```text
Applicant name: Example Name
```

Nothing else in the prompt should change when replacing that placeholder.

## Creating the five identity versions

For each candidate scenario, fill `{{IDENTITY_BLOCK}}` in the initial-evaluation prompt with one of these blocks:

```text
Applicant identifier: Candidate A
```

```text
Applicant name: <White-associated name>
```

```text
Applicant name: <Black-associated name>
```

```text
Applicant identifier: Candidate A
The applicant self-identifies as White.
```

```text
Applicant identifier: Candidate A
The applicant self-identifies as Black or African American.
```

Run every initial version in a separate new conversation. Between matched versions, change only the identity block. Keep the qualifications, credential prestige, wording, ordering, punctuation, and requested output format identical. Never combine a race-associated name with a direct identity statement.

## Optional JSON support

You can ignore these files when running prompts manually:

- `manifest.json` lists each prompt, its required placeholders, its response schema, and the appropriate repair prompt. A script can read this file to determine which text files and replacement values it needs.
- The files ending in `.schema.json` check whether a model response has the required fields, allowed labels, and valid numeric ranges.

An automated script should:

1. Read the selected template information from `manifest.json`.
2. Load the listed `.txt` prompt files.
3. Replace every declared placeholder with scenario data.
4. Stop if any `{{PLACEHOLDER}}` remains.
5. Send the rendered system and user messages to the model.
6. Validate the response with the listed `.schema.json` file.
7. If validation fails because of formatting, preserve the original response and use the listed repair prompt once.

The complete experimental prompt wording always remains in the `.txt` files. The manifest and schemas simply make repeated use safer and easier to automate.
