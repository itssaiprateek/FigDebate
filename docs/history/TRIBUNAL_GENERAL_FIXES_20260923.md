# Tribunal general fixes and qualification — 23 September 2026

Status: engineering fixes implemented and focused qualification completed. The experimental semantic audit is excluded because it regressed a helpful control. This document does not certify dataset-wide accuracy or zero failures.

## Baseline and scope

Five random diagnostic batches contain 50 distinct cases, with 28 initially correct and 27 finally correct. There were 12 label-correct and 10 label-wrong change proposals, one helpful acceptance and two harmful acceptances. Correct proposal labels do not establish correct explanations. The tribunal used the compact proposer, V5 verification, bounded repair, baseline audit and disabled feedback.

The initial arbiter, upstream model choices, dataset labels and feedback mode remain unchanged. Before-edit source is preserved under `research/tribunal_general_20260923/before` outside this repository. Existing unrelated working-tree edits are preserved.

## Research review and fit

The previous [31-source method register](../../../TRIBUNAL_SOURCE_REVIEW_20260922.md) remains the core bibliography. This review also inventories 143 distinct displayed title entries in the longer reference panes of all four supplied research attachments in [supplied_titles.txt](../../../research/tribunal_general_20260923/supplied_titles.txt). Those panes contain truncated titles, duplicates, mirrors, news, author pages and conference indexes; they are not 143 independent papers. Screening an entry for relevance is not the same as reading its full method. Unresolved titles are not used as implementation evidence.

Hugging Face's paper-search connector returned `Tool paper_search not found`. Direct Hugging Face pages and primary arXiv/ACL pages were used instead. Most promising mechanisms were checked in primary text; additional background papers were screened through primary abstracts. No cited paper's headline gain is a predicted FigDebate gain.

| Method / source | Fit and decision |
|---|---|
| [DynaDebate](https://huggingface.co/papers/2601.05746), [primary method](https://arxiv.org/html/2601.05746v2) | Localize a bad inference instead of rewarding fluency or agreement. Test a compact step-focused audit using the existing call. Its diverse path generation and external tool results are not reproduced here. |
| [MARCH](https://huggingface.co/papers/2603.24579), [primary method](https://arxiv.org/html/2603.24579v1) | Keep independent evidence selection and relation checking blind to the proposer. MARCH additionally trains agents with MARL; prompt separation alone is not that result. |
| [AgentAuditor](https://huggingface.co/papers/2602.09341) | Evidence-focused checks at reasoning divergences support localized repair. Full reasoning-tree search and anti-consensus preference training would add unqualified complexity and data requirements. |
| [SymDiag](https://arxiv.org/html/2608.08786v1) | Separate a defective representation/audit from a defective candidate. Its actual backend is SWI-Prolog. Do not claim formal verification of figurative meaning from JSON consistency. |
| [VERGE](https://huggingface.co/papers/2601.20055) | Distinguish formalizable consistency checks from soft semantic judgments. Exact source binding is deterministic; inferred mood, intention and metaphor still require fallible interpretation. |
| [Fact in Fragments](https://arxiv.org/html/2506.07446v1) | Necessary-condition decomposition can improve evidence coverage, but atomization must preserve expressions and scope. The earlier richer premise schema lost a helpful control; do not reinstall it merely because the concept sounds suitable. |
| [The Format Tax](https://arxiv.org/html/2604.03616v1), [Capacity, Not Format](https://arxiv.org/abs/2606.09410) | Separate genuine incomplete generation from forced character closure. Remove redundant prose ceilings while retaining finite token/context/time budgets. Delayed formatting is a future controlled experiment, not an assumed cure. |
| [TruncProof](https://arxiv.org/abs/2605.13076), [XGrammar 2](https://arxiv.org/abs/2601.04426) | Grammar completion is transport protection, not evidence of semantic completeness. No unqualified decoder migration or automatic closure of incomplete semantic output. |
| [Stay Focused](https://arxiv.org/abs/2502.19559), [MAR](https://arxiv.org/html/2512.20845v2) | Repeated critique can drift. Preserve the original source and candidate during a tribunal-only repair; do not convert a verifier opinion into a new proposal. Neither paper proves a universal prompt-only cure. |
| [Trust but Verify / PVD](https://arxiv.org/abs/2605.25133) | Bounded targeted challenge is relevant. Acceptance-without-revision is an empirical signal, not a soundness theorem; the paper itself reports failures with inadequate verifier competence. |
| [Knowing When Not to Answer](https://arxiv.org/html/2604.14799v1), [AgentAbstain](https://huggingface.co/papers/2607.10059) | Measure unnecessary abstention separately from justified uncertainty. Do not force binary judgments to improve an acceptance count. |
| [SCOPE](https://arxiv.org/html/2602.13110v1), [learned conformal abstention](https://arxiv.org/abs/2502.06884), the three conformal sources in the core register | Need separate representative calibration and, for the learned policy, training. A conformal p-value is not the probability a proposal is correct. No thresholds fitted on these 50 examples. |
| [Cross-style multimodal reasoning](https://arxiv.org/abs/2601.17197), [V-FLUTE](https://aclanthology.org/2025.naacl-long.1/) | Strong evidence that task-specific semantic learning merits a separate experiment. Preserve the entailment task, not merely sentiment similarity. No test-set reference explanations enter inference or training. |
| [AffectAgent](https://arxiv.org/abs/2604.12735), [IDRAAK](https://arxiv.org/abs/2608.08801) | Screened for emotion reasoning and semantic drift. Jointly trained multimodal agents and requirements reconciliation in another domain are not validated prompt-only replacements for this tribunal. |
| [From Spark to Fire](https://arxiv.org/abs/2603.04474), [Collective Hallucination](https://arxiv.org/abs/2606.07941), [Hallucination Cascade](https://arxiv.org/abs/2606.07937) | Preserve provenance and distinguish new observations from recycled criticism. Network defenses and confidence weighting are not proof of figurative truth. |
| [Delayed Verification](https://arxiv.org/html/2606.27409v1), [Koopman certification](https://arxiv.org/abs/2608.05956), ASI and spectral-drift sources in the core register | Convergence/dynamics results do not certify our answers. No evidence that these abstract dynamics explain our synchronous GPU timeout. |
| [Diversity Collapse](https://arxiv.org/abs/2604.18005), AceMAD, Debate or Vote and identity-bias work in the core register | Preserve independently formed judgments; do not use agreement as truth. Scoring, topology and anonymization interventions need their own comparisons. |
| [MACM](https://arxiv.org/abs/2404.04735), [PACER](https://arxiv.org/abs/2602.02828), [CoVER](https://arxiv.org/abs/2606.09064) | Condition mining, localized revision and evidence acquisition are useful analogies. Math ensembles or long-video retrieval are not drop-in solutions for a single figurative image. |
| [VerifyMAS](https://huggingface.co/papers/2605.17467), [iterative audit case study](https://arxiv.org/abs/2605.12280) | Inspect whole trajectories and cross-file contracts. Neither eliminates the need to reproduce each suspected implementation defect. |
| [Structured-output benchmark](https://arxiv.org/abs/2501.10868), [software structured-output study](https://arxiv.org/abs/2606.09395) | Relevant evaluation background. Schema success and reasoning success must be reported separately. |
| [Safe remediation](https://arxiv.org/abs/2607.20005), [learning escalation](https://arxiv.org/abs/2601.07006) | Learned intervention policies require task-specific logs/calibration. Microservice repair or moderation thresholds cannot be imported into this gate. |
| R2-MAD, episodic memory, precedent retrieval | Deferred with feedback disabled. Any future memory needs independent training provenance and evaluation leakage checks. |
| Trading, legal, cybersecurity, agricultural, time-series, software-agent, general coordination and survey entries in the supplied panes | Screened as background or different application domains. No domain-specific mechanism is copied into the image entailment pipeline without a concrete mapping and evidence. News/index/mirror pages are not separate supporting experiments. |

The paper-inspired semantic pilot changed only the audit rubric; it added no agent, vote, confidence threshold or extra call. It remains outside production because the paired controls demonstrated a regression.

## Implemented mechanisms

1. **Explicit absence is different from uncertainty.** The audit accepts the literal spelling `None` only when `alternative_status=NONE` and `alternative_relation=UNRESOLVED`. The raw response is retained. `unknown`, a real alternative, inconsistent relations, objections and missing deciding citations still fail.
2. **Complete text survives transport.** The semantic bridge no longer slices text at 1,200 characters. V5 mapping and relation schemas no longer force prose or source quotations to end at 240/480 characters. The corresponding validators use the same schema. Finite output-token, context and deadline bounds, exact source spans, citation constraints and unfinished-clause checks remain.
3. **Criticism cannot rewrite the candidate.** Eligible tribunal-only disagreement, uncertainty and audit repairs freeze the original compact candidate and recheck the disputed proof. A new witness hearing can still supply evidence for a fresh proposal. Frozen repairs do not reserve a nonexistent proposer call. Successful prerequisites are reused only under exact input-bound cache keys.
4. **Audit transport is checked against raw output.** V5 challenges must exactly match their recorded full response or their explicitly recorded field repair. Plain-JSON relation responses now have the same readback identity requirement as indexed-source responses. Mutated explanations, omitted objections or overwritten judgments cannot acquire a certificate merely by retaining a `verified` flag.
5. **An auxiliary family label is not an extra semantic veto.** V5 directional admissibility comes from its case-bound relation and argument checks. Changing only a coarse bridge-family label no longer rejects an otherwise complete proof. Missing/contradictory evidence, unresolved audits and failed provenance remain blocking. Legacy protocols keep their existing policy.
6. **A semantic contradiction is not a formatting failure.** A complete V5 relation response that claims SUPPORT while retaining a necessary unresolved condition is recorded as completed execution, valid structure and contradictory semantics. It is not regenerated under a FORMAT REPAIR instruction, cached as successful evidence or passed to the final challenge. Existing bounded semantic repair remains available when there is sufficient budget. Actual schema failures retain their existing recovery path.
7. **A follow-up needs a budget for its whole remaining proof.** The scheduler uses the slowest completed attempt in the current case, with the existing 30-second stage floor, to reserve time for all remaining uncached obligations. It does not learn from preceding samples or raise the case deadline. This is a scheduling estimate, not a guarantee against future slowdown.

These are shared mechanisms. No production rule contains a sample ID, image hash allowlist, caption match, expected label or phenomenon-specific acceptance exception.

## Pipeline audit and remaining limits

| Stage / issue | Finding and disposition |
|---|---|
| Dataset and initial arbiter | Kept unchanged. Gold labels/reference explanations remain evaluation-only. |
| Factual and caption witnesses | Capability routing and per-question identity checks are active. The visual witness receives an image and an observable question; the caption witness receives the complete caption. Unequal numbers of questions are allowed when different capabilities are needed. |
| Legacy prompt builders | Some old functions contain character limits, but the active caption witness uses typed `audit_claim_witness`, and the active visual adapter extracts the atomic question. Do not mistake unreachable legacy prompts for the current missing-information cause or change the initial pipeline unnecessarily. |
| Proposer | Compact four-field structure retained. Source, negation, modality, comparison and role errors remain possible model mistakes. |
| Evidence selection | Full literal catalogue is available; selected evidence can still omit a decisive fact or mark coverage incorrectly. Citation existence is not semantic coverage. |
| Mapping | Exact quotation and identity checks cannot establish metaphor equivalence. Complete expressions remain available downstream. |
| Relation | Positive incompatibility is required; shared absence-as-conflict mistakes remain possible. No expanding topic/keyword exception list is used as a purported solution. |
| Condition bookkeeping | Existing overlap/contradiction checks remain. A future single authoritative condition table should be compared live before replacing this representation. |
| Challenge | Placeholder handling fixed. Substantive false objections and self-contradictory semantic judgments remain fallible model outputs. |
| Gate | Acceptance still requires complete, case-bound evidence, matching relation and resolved material audit. Correct labels with unsound reasons are not automatically accepted. |
| Repair | One bounded hearing, independent of disabled feedback; cached obligations remain image/source/prompt/schema/model/profile bound. Generic disagreement cannot rewrite a candidate. Contradictory semantic outputs cannot be cached as successful evidence. |
| Timing | No arbitrary timeout increase. GPU clock drops were observed previously, but their cause was not established. Finite budgets cannot guarantee completion under unbounded slowdown. |
| Recovery | Existing one-retry parsing and context checks remain for actual format/transport failures. Complete semantic contradictions leave that retry path. Partial outputs are not relabelled as successful abstentions or certificates. |
| Export/resume | Atomic records, exact run identity, checkpoint and export recovery stay in the common path; focused tests exercise these without dataset inference. |
| Data specificity | Searches and call-path inspection found no active ID/caption/gold-specific production branch. Generic English routing and relation heuristics still have limited semantic coverage; this is not a proof of universal correctness. |
| Reporting | Label correctness, explanation quality, execution failures, harmful acceptance and wall time must remain separate. |

## Qualification

Final focused regression suite: **176 tests passed**, including nine new adversarial/generalization tests. These exercise explicit absence versus uncertainty, long source/prose preservation, incomplete outputs, frozen disagreement repair, renamed/reordered evidence identities, raw-record mutation, family-label invariance, semantic-versus-format classification and case-local repair budgeting. Raw-record mutation and family-label defects were reproduced as failing tests before their fixes. Existing tests cover routing, feedback-disabled scheduling, context limits, interrupted generation, checkpoint identity and failed-export recovery. A simulated 6,000-case plumbing test checks bounded image residency; it performs no model inference.

Saved-output replay, with no inference: baseline 28 initially correct -> 27 final; after mechanical revalidation 28 -> 28. Helpful accepts 1 -> 2; harmful accepts remain 2. The change recovers the existing complete correct audit blocked by the spelling `None`; it does not regenerate missing proofs or show that all newly generated judgments improve.

The initial arbiter's source hashes match the before-edit snapshot. The final gate replay still produces 28/50 final correct, two helpful accepts and two harmful accepts; no replayed harmful acceptance was added by these changes.

The localized audit pilot is **not retained**. Across four paired frozen-proof checks, baseline accepts were two helpful and one harmful; the candidate retained only one helpful and blocked the harmful one. However, its harmful rejection was an incomplete self-contradictory audit, and it also blocked the helpful therapy control. It did not resolve the fertilizer correction. A rejection caused by malformed bookkeeping is not evidence of a correct semantic detector.

The bounded live pilot comprised eight final-audit evaluations (four paired comparisons, each subject to the existing retry allowance) and two fresh compact-tribunal reviews through the actual gate, with saved upstream testimony. It did not rerun the upstream agents or a full dataset batch. The two fresh reviews produced one correct acceptance in 232.12 seconds and one rejected correct-label proposal with a final-audit timeout in 241.78 seconds. Neither result establishes the quality of its explanation across other cases.

The failed live control identified the last engineering fix. Its first relation response completed in 50.39 seconds, with 144 generated tokens, no token-limit hit and no character-boundary hit. It asserted SUPPORT and simultaneously retained an unresolved caption condition. The wrapper classified that semantic conflict as invalid formatting and issued a full-prompt retry. The two attempts consumed 131.17 seconds in the relation stage, leaving about three seconds for the challenge, which timed out. The second response also dropped the unresolved condition, demonstrating why a formatting-repair instruction is the wrong place to resolve a semantic dispute.

After the fix, the exact captured visual, mapping and first-relation bytes were replayed through production validation and the gate. This replay used a stub image with a consistent test identity: it tests response transport and control flow, not image understanding. It made exactly three verification calls, preserved the unresolved text, classified the relation as `COMPLETED / VALID / CONTRADICTORY`, performed no format retry, rejected the proposal and retained the initial answer. Using captured durations, about 82.52 seconds remained versus a 100.78-second reserve for the remaining proof, so no follow-up started. These figures are counterfactual scheduling estimates, not a new live speed measurement.

**The final retry/budget fixes have recorded-response and regression qualification, but no fresh live model rerun after those last edits.** The live timeout remains part of the evidence; it is not erased or counted as successful reasoning. GPU clocks were unusually low during the live checks, but no OOM occurred, and this review did not establish the cause of the clock behavior. Broader timeout-free operation remains unproven.

All ten captured live results were rechecked under the final gate and retained their recorded acceptance decisions. The separate 50-case saved-output replay remains 28/50 final correct, with two helpful and two harmful accepts. This validates compatibility of those saved outputs; it is not an end-to-end result from newly generated outputs.

## Exact scope and evidence

Eight existing production files changed during this task:

| File | Retained change |
|---|---|
| `engine/tribunal_process.py` | Shared, strictly typed handling of an explicit no-alternative value. |
| `engine/evidence_review_v5.py` | Matching text schemas/validators, raw-response integrity and semantic-contract rejection. |
| `engine/evidence_verification.py` | Optional version-specific relation schema; completed semantic conflicts do not trigger format retries or enter the successful cache. |
| `engine/review_outcome.py` | Preserve the semantic error in the typed stage report. |
| `engine/semantic_bridge.py` | Remove silent 1,200-character source/bridge clipping. |
| `engine/semantic_bridge_verifier.py` | V5 proof controls directional admissibility, rather than an auxiliary family label. |
| `engine/simple_judge.py` | Reuse the frozen candidate for the shared set of tribunal-only repair reasons. |
| `engine/tribunal_repair.py` | Frozen candidate scheduling, exact-input cache reuse and current-case whole-proof budget estimates. |

The new focused tests are in `tests/test_tribunal_generalization.py`. This task's [source diff](../../../research/tribunal_general_20260923/this_task.patch) is against the pre-edit snapshot, so it excludes unrelated earlier working-tree changes. Evidence is saved in [regression.log](../../../research/tribunal_general_20260923/regression.log), [saved-output replay](../../../research/tribunal_general_20260923/replay_after.json), [captured failure replay](../../../research/tribunal_general_20260923/timeout_replay.json), [live records](../../../research/tribunal_general_20260923/live_20260923_230546/records.jsonl) and [final verification](../../../research/tribunal_general_20260923/final_verification.json).

The remaining research problem is still substantive: independent stages can select incomplete facts, misunderstand figurative scope, and agree on an unsupported explanation. Stricter fields, a bigger prompt or agreement among the same model's calls do not establish truth. The retained fixes remove demonstrated transport, contract and scheduling defects; they do not solve those shared semantic errors. Any next semantic replacement needs a paired comparison that retains helpful corrections as well as reducing harmful acceptance, followed by the user's broader dataset evaluation.
