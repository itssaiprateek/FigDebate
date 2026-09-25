# Team handoff

September 25 evaluation addition: [offline qualification diagnostics](QUALIFICATION_DIAGNOSTICS.md)
now reports explanation delivery, recorded error origins, and correction outcomes.
It is separate from inference and does not change any arbiter, tribunal, OCR, or
acceptance behavior. Experimental semantic changes remain in the research workspace
until they pass qualification.

## Current release scope

The user explicitly approved adding the OCR retention candidate after reviewing
its demonstrated benefit and incomplete end-to-end qualification. It is now in
the main source. This approval is distinct from a passed semantic release gate.

Only two inference settings changed in `agents/visual_adapter.py`:

- OCR word ceiling: 180 to 360, retaining completed long crop readings.
- Short presence-answer ceiling: five to six total words, matching the existing
  prompt's YES/NO/UNCLEAR plus five factual words.

Generation budgets, prompts, truncation/EOS checks, repeated-output checks and
other validation rules remain unchanged. Every other existing Python source file
is checked against the pre-cleanup manifest. Runtime modules were not renamed or
removed. Existing arbiter, tribunal, repair, model adapters and acceptance behavior
are preserved. The new regression file is `tests/test_visual_answer_retention.py`.

The offline BERTScore evaluator from the preceding work is included, with an
optional `requirements-evaluation.txt`. It reads human references after inference
and cannot change model decisions. See [evaluation guidance](REASONING_SIMILARITY.md).

## Evidence and limitations

| Check | Result |
|---|---|
| Six-case visual-stage comparison | Two previously discarded completed OCR readings recovered; no newly invalid visual answers. Mean completion 0.95455 to 0.98485. |
| Dense document, fresh visual stage | OCR delivery 0 to 374 words; completion 0.7273 to 0.9091. Truncated whole-page reading still rejected. |
| Complete paired downstream comparison | Interrupted at user request; candidate final answer unavailable and six preselected validation pairs not run. |
| Semantic variants | Mapping corrected one label but missed the wordplay; other tested tribunal variants yielded no accepted corrections. None is promoted. |
| Main-source software suite after OCR adoption | 681 tests; suite passed, four existing skips and one expected failure. |
| Extracted source-only handoff package | Same 681-test suite passed, with the same skips and expected failure; Python source hashes match the final handoff. |

The OCR change fixes evidence delivery. Improved semantic accuracy or accepted
corrections has not been established. Remaining defects include inferred roles,
intent versus actuality, negation and temporal polarity, figurative interpretation,
and incomplete initial explanations retained when the tribunal abstains.
See the [system-origin audit](history/SYSTEM_ERROR_ORIGINS_20260924.md) and
[completed experiment report](history/RESEARCH_IMPLEMENTATION_QUALIFICATION_20260924.md).
Those reports describe the state before this subsequent user-approved adoption.

## Navigation and operation

- [RUNBOOK](RUNBOOK.md): environment, assets, software checks and development run.
- [ARCHITECTURE](ARCHITECTURE.md): responsibility-to-module map.
- [Documentation index](README.md): current contracts and historical reports.
- `run_figdebate.py`: official pipeline entry point.
- `scripts/run_development_checks.py`: portable ten-case development launcher.
- `run_compact10.ps1`: compatibility wrapper; no personal Python/model paths.

The handoff commands explicitly disable feedback. Existing CLI defaults and
alternative modes were not rewritten. Use new output directories for new code;
the source-integrity guard deliberately prevents resuming old-source runs as if
they used the current OCR behavior.

## Cleanup and retained history

Fifty older reports/records now live in `docs/history/`, with an index and updated
resolvable Markdown links. The old README is retained there as historical context.
Current documentation no longer mixes old trial instructions with the runbook.
Historical tests and ablation code remain because they protect working behavior.

Nine failed or unqualified experimental source variants were archived outside
the repository under `research/adoption_20260924/archives/failed_source_variants/`:
audit, delivery, delivery_budget, grammar, grammar_ordered, mapping, region,
region_grammar and repair. Every source-file hash was verified against its ZIP
before the loose copy was removed. Raw results and protocols remain available;
failed results have not been erased or counted as successes. The baseline and
OCR candidate snapshots remain for reproducibility. Restore an archived source
to its original research `sources/` location before replaying an old trial.

Local runs are kept on disk but ignored by Git. The source-only handoff archive
excludes environments, caches, model weights, processed datasets, local runs,
editor settings and failed experimental source copies. Teammates obtain dataset
and model assets separately using the runbook. The archive contains the current
working source, including earlier uncommitted pipeline work; it is not presented
as a clean historical commit. No working behavior was reverted to the last commit.

## Next validation for the receiving team

Run a matched baseline-versus-OCR comparison from images to final answers on the
preselected validation cohort and the dense-document regression, using preserved
source versions and new output directories. Keep references out of inference.
Report accepted helpful/harmful changes, missing/truncated answers, context and
runtime failures, reference similarity and human-reviewed reasoning separately.
Do not turn development cases into production rules or claim a general gain from
the recovery of two OCR readings.

The pre-cleanup hashes, software-test log, archive manifest, launcher preflight
and package verification are recorded outside the repository in
`research/handoff_20260924/`. The final verification summary is included as
`docs/HANDOFF_VERIFICATION.json` in the handoff package.

Rebuild a source-only package with `python scripts/build_handoff.py --output dist/FigDebate_team_handoff.zip`.
Use a new filename if that archive already exists. Every included source file is
checked against `SOURCE_MANIFEST.json` inside the archive. Historical reports can
refer to raw research/run artifacts that are retained separately and not bundled.
The repository-wide whitespace check still reports the two preexisting blank-EOF
findings listed in the verification file; their working source was preserved.
