"""Repository-relative paths.

All paths derive from the repository root so nothing depends on where a run is
launched from. The root is located by walking up from this file rather than by
an environment variable, because the data and prompt files are versioned
alongside the code and must never be read from an unversioned copy.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
PROMPTS = ROOT / "prompts"

JOB_SPECS = DATA / "job-specs"
SCENARIO_SOURCE = DATA / "generated-candidate-scenarios"
STIMULI = DATA / "stimuli"

OUTPUT = ROOT / "output"
