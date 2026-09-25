# FigDebate

FigDebate evaluates how an image supports or contradicts a figurative caption,
including humor, metaphor and sarcasm. Visual and language agents produce evidence
and a claim interpretation; an arbiter proposes a decision, and a tribunal can
review it through independent checks and a deterministic acceptance gate.

**Team handoff:** start with [HANDOFF](docs/HANDOFF.md), then follow the
[RUNBOOK](docs/RUNBOOK.md). [ARCHITECTURE](docs/ARCHITECTURE.md) maps the code.

## Current status

The approved OCR retention fix is enabled. Completed crop readings up to 360
words are retained; truncated free-form responses are still rejected. Short
presence answers permit YES/NO/UNCLEAR plus five factual words.

This fixes demonstrated evidence loss. It does **not** establish improved
semantic accuracy or tribunal acceptance: the complete paired validation was
interrupted. The other research variants are not integrated. Feedback remains
disabled in the supported handoff commands.

## Entry points

| Task | Entry point |
|---|---|
| Official pipeline | `run_figdebate.py` |
| Fixed development cohort | `scripts/run_development_checks.py` |
| Software regression suite | `python -m unittest discover -s tests` |
| Human-reference similarity | `python -m evaluation.reasoning_similarity --help` |
| Existing compatibility launchers | `figdebate.py`, `run_reproducible.py`, `run_compact10.ps1` |

```powershell
# After configuring the environment and dataset as described in the runbook:
python scripts/run_development_checks.py --selection-only
python -m unittest discover -s tests
```

The selection-only command does not run model inference. Remove that option for
the ten-case development check; it is a diagnostic cohort, not a held-out estimate.

## Repository layout

```text
agents/        Visual evidence, caption interpretation and witness interfaces
arbiter/       Initial decision and existing arbitration behavior
comparators/   Evidence comparison
engine/        Orchestration, ledgers, tribunal, contracts and acceptance
models/        Model adapters and pinned source definitions; no bundled weights
dataset/       Dataset preparation, loading, selection and provenance
evaluation/    Decision, evidence and explanation evaluation
config/        Explicit development IDs and existing protocol configuration
scripts/       Portable operational launchers
tests/         Regression tests, including prior mechanism fixtures
docs/          Current handoff, runbook, architecture and evaluation guidance
docs/history/  Previous reports; historical commands are not current defaults
outputs/       New local runs (excluded from Git and handoff archives)
```

Existing runtime module and test names remain stable to preserve imports and
regression coverage. Local `runs/`, weights, environments, caches and failed
experiment source copies are excluded from the source handoff package. See
[the handoff record](docs/HANDOFF.md) for evidence, limitations and archive policy.
