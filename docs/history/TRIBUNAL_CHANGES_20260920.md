# Tribunal reliability fixes and experimental process audit

The 20 September work separates engineering validation from semantic qualification. The initial visual agent, caption agent, comparator and arbiter implementations are unchanged. Feedback remains disabled. No production behavior is keyed to a dataset example, caption or reference answer.

## Available behavior

- V5 citation schemas now require the citations already required by their validators. An entirely unmatched mapping can be reported honestly with no invented binding; it remains ineligible for acceptance.
- Review traces identify individual execution/validation/semantic stages and the first observed blocking stage. CSV exports include these fields. Unexecuted downstream steps are not reported as additional failures.
- `--tribunal-repair-mode bounded` enables the existing bounded V5 semantic-repair path independently of feedback. It is opt-in; the default is `disabled`. There is no additional semantic round beyond the existing two-round ceiling, no reset of spent budget and no relaxation of the gate.
- Internally contradictory alternative/status fields now have a specific recheck route. Software does not normalize these fields into an accepted proof.
- `--tribunal-audit-mode process-audit-1` enables the unqualified experimental source-task and process-audit variant. The default remains `baseline`. Both nondefault options require stagewise tribunal execution, enabled hearings, the explicit V5 profile and disabled feedback.

The experiment preserves full source expressions alongside focused quotations and asks the existing final audit to check interpretation, proposition alignment, necessary-condition coverage and inference. Its extra audit fields carry citations and reasons; they are not trusted as Boolean truth certificates. Failed or unresolved process checks cannot authorize acceptance. The candidate audit has a 768-token allowance versus 384 for the baseline audit, within the unchanged per-case limits; it is therefore a configuration experiment, not a matched-compute prompting ablation.

Missing-information relevance can be reconsidered within the existing semantic repair. If an external fact is unavailable, the tribunal may examine whether it is necessary; no witness is asked to invent it. Material or unresolved uncertainty remains an allowed outcome. Hypothetical alternatives never become evidence.

## Qualification status

Engineering paths have focused regression coverage. Experimental semantic usefulness has not been established. The initial paired Qwen check covered three development cases in two arms, with frozen upstream agents and no new live witnesses. Neither arm corrected an answer; neither changed an initially correct answer to a wrong one. The process-audit candidate is not recommended as a default or as justification for a large evaluation.

The separate graph controls deliberately script earlier proof stages and run the final audit live. They test a narrow mechanism, not figurative-language accuracy. Their rechecks are diagnostic evidence and do not replace fresh end-to-end qualification.

Artifacts and the detailed completion report are in the enclosing workspace's `research/tribunal_changes_20260920/`. The standing instruction remains: `--feedback-mode disabled`. Do not claim paper readiness, dataset-wide improvement, zero semantic errors or beneficial repair from code tests alone.
