# Tribunal implementation and qualification — 17 September 2026

The proposed C1–C6 mechanisms are implemented. The new semantic review remains an **unqualified, opt-in candidate**. The targeted live checks have not established an accuracy improvement. Do not start a 6,000-sample evaluation of this candidate on the assumption that it is ready.

The original arbiter, visual/claim extraction, comparator, initial debate logic and dataset code were checked against the exact dirty source snapshot saved before this implementation: all 19 protected files match. No production rule uses a sample ID, example caption, image hash as an answer key, reference label or expected answer. Hashes identify inputs and prevent reuse across changed cases. Evaluation references stay outside model inputs.

## Implemented changes

| Area | Change | Availability |
|---|---|---|
| C1: reproducibility | Save the exact source archive and hashes, including uncommitted files. Distinguish failed execution, semantic uncertainty, rejected proposals and verified conclusions. Report corrections, harms, repair contribution and time separately. | General infrastructure |
| C2: consistent decision task | Preserve the original assertion, participants, qualifiers, scope, and intended versus actual outcomes across the proposal and verification calls. Missing evidence cannot establish contradiction. | Review 5 candidate |
| C3: independent selection | The verifier selects up to four deciding records from the current image and full eligible literal catalogue without seeing the initial/proposed label. Proofs bind to that catalogue, selected evidence and original inputs. | Review 5 candidate |
| C4: acceptance | Require a complete source-bound proof, role mapping, independent relation and a challenge grounded in deciding evidence. Constrain citations to selected IDs. A confidence value or agreeing flag cannot bypass the gate. | Review 5 candidate |
| C5: feedback | Replace standalone precedent retries with one diagnostic repair inside the existing two-hearing limit. Route observable questions to vision, caption questions to language, and disputed inferences to the tribunal. A repaired answer must pass the same gate. | Review 5 with integrated feedback |
| C6: bounded completion | Generate comparisons once; copy source spans and repeated observations in software. Repair a malformed generated field while revalidating retained fields. Reuse source-bound obligations when unchanged. Freeze nominal review budgets instead of borrowing earlier cases' cost history. | Review 5 candidate |
| Execution | Restore spent case budget across a resumed/second hearing. Detect Windows host suspension and discard partial judge output. Retain existing 120-second judge generation and 240-second cumulative judge-work ceilings; model loading and witness stages are additional. | Shared execution infrastructure; fixed allocation is Review 5 only |
| Larger runs | Decode at most 32 images per batch by default. Keep final traces on disk, retain only result identities/digests in memory, and stream exports. Preserve immutable commits and reject conflicting/corrupt records. | Stagewise runner and exports |

A concrete routing error was fixed: the prohibition on asking a visual witness to decide contradiction also suppressed semantic contradiction questions intended for the tribunal. Such questions now reach the tribunal; requests for ground truth or final labels remain blocked. Unavailable or repeated requests still stop, and failures are not counted as successful semantic decisions.

## Compatibility and use

Existing `paper-8gb`, `8gb`, `12gb`, `16gb` and automatic settings retain review protocol 4.0. The previous four normal verification prompts, token budgets and scripted proof audit were compared directly with the saved baseline and match. This is structural preservation, not a claim of identical GPU numerical results.

The candidate is explicitly selected with `--hardware-profile paper-8gb-review5`. Its image resolution, model and hardware ceilings match `paper-8gb`. `--feedback-mode integrated` enables the single diagnostic repair; `--feedback-mode disabled` supplies the repair-off ablation. Integrated feedback rejects a noncandidate profile, and the candidate rejects legacy precedent mode. These are configuration details, **not a recommendation to launch a large evaluation yet**.

`--batch-size 32` controls decoded image residency. Raw dataset storage, model memory and saved disk traces still require resources; the synthetic check does not prove that a full model run fits every machine. Batch size and source identity are recorded. Use a fresh run directory after source changes rather than mixing results into an older evaluation.

## Verification

- Final targeted regression suite: **171 tests, 170 passed, one pre-existing expected failure**. The expected failure is the unchanged arbiter test `test_scoring_cannot_condition_on_assessment`; it is not an unexpected regression introduced here.
- Scripted proofs pass the real acceptance gate in both label directions. Changed images, captions or evidence invalidate the proof. These tests establish software behavior, not model correctness.
- A synthetic 6,000-record iterator retained at most 32 decoded images; owned images were closed after batches and on decode failure. This used no model inference.
- Disk-backed results, integrity failures, legacy journal migration, checkpoint recovery and streaming exports passed targeted checks.
- No full development or 6,000-sample model run was launched. Detailed live results and limitations are recorded in the accompanying research report.

## Qualification boundary

The live model still produces slow or inconsistent reasoning. A timeout that preserves an initially correct label is **not** evidence that the gate understood and rejected a harmful proposal. Blocking every proposal also fails the objective: useful corrections must survive.

For that reason the semantic candidate is implemented but not promoted to the normal settings. It needs completed helpful corrections without added harms on an appropriate mixed panel, followed by a disjoint, frozen evaluation. Integrated feedback needs a repair-on versus repair-off benefit; an attempted second hearing alone does not qualify it. Do not lower acceptance thresholds or add example-specific instructions to make the development cases pass.

If the completed, compact checks remain semantically unreliable, the next bounded experiment is a verifier/model capability comparison with frozen inputs and gate rules. Further prompt/schema expansion is not justified by the present evidence. Paper readiness and generalization across approximately 6,000 examples remain unproven.

Detailed artifacts are in `research/tribunal_implementation_20260917/` in the enclosing workspace, including the before snapshot, source audit, unit logs, source archives and live case traces.
