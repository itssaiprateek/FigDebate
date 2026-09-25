# Integrated tribunal repair — 7 September 2026

Latest critical-repair status: [implementation and qualification](CRITICAL_REPAIR_IMPLEMENTATION_STATUS_20260907.md).
The 24-item research proposal is **not fully implemented**; that status file
distinguishes integrated code, diagnostic results and remaining graph/research work.

Implementation and qualification remain in progress. **No substantial accuracy
gain, calibrated confidence or publication readiness is claimed.** All 116
requirements remain in REQUESTED_CHANGE_LIST.md; change_tracker.json maps each
to source evidence and unfinished validation. Statuses do not count empirically
successful changes.

## Preserved constraints

- Actual code: C:/Users/Sai Prateek/Desktop/FigDebate_sent/FigDebate_main,
  branch semantic-bridge-v3, baseline HEAD f96abdca79f06988a9128d432c9aeed4fc1d996e.
  Existing dirty work was preserved. No commit or destructive reset was made.
- Dataset stays humor, metaphor and sarcasm. No stored split, image or label was
  rewritten. Clean upstream reconstruction awaits user-arranged Hugging Face access.
- Raw caption immutable; gold/reference explanations remain evaluation-only.
- Existing pinned quantized Qwen3-VL 4B, Mistral 7B and Qwen3.5 4B remain in use;
  RTX4060 Laptop 8GB profile. No larger model was downloaded.

## Engineering progress in this continuation

1. Primary accuracy now counts all prediction-file rows, including missing
   outputs and binary answers whose evidence contracts fail. No-answer confusion
   counts, separate contract coverage and incomplete-run disclosure prevent
   successful-only reporting.
2. Independent-vote and two-round conventional-debate controls share the canonical
   runner/API. Failed/unresolved agents cannot shrink the majority denominator.
   Voting results never inherit baseline evidence authority or confidence.
3. Executed generation requests/tokens are recorded for vision, claim, candidates,
   arbiter, judge and retries. Cached work is distinct. Unknown failed-attempt
   tokens stay unknown; scoring-forward computation is explicitly outside scope.
4. Typed caption-only hearing requirements use bounded semantic repair. Witness
   inputs retain roles, negation, quantities, modifiers and scope. Lexical
   heading repair no longer certifies witness semantics.
5. Judge clause-saturation repair, complete-JSON stopping and a compatible
   MIT-attributed constrained-decoding adapter address output-contract failures.
6. Fresh image checks are bound to actual decoded pixels and raw caption;
   two genuinely order-swapped calls do not imply independent model errors.
7. A bounded omitted-evidence retrieval/re-review cycle exposes requested evidence
   before reliance. Only final-packet evidence can be cited.
8. Shared provenance roots and exact duplicate root content are not counted as
   independent corroboration. Explicit panel/scope distinctions remain. Final
   traces expose admissible ancestors and disclose incomplete evidence chains.
9. Locked internal partition selection, separate score-calibration and blinded
   image-preserving two-rater packet/agreement tooling exist. Calibration now
   reads the actual exported final_confidence field.
10. All 116 tracker entries have evidence/limits. Ten papers / 40 saved source
    files have verified hashes. METHOD_ADAPTATIONS.md distinguishes local
    adaptations, licenses, prior art and unproven transfer to FigDebate.

## Validation evidence

- Full regression suite: **321 tests passed**, including the live-discovered
  fully cached runner and invalid-witness filtering regressions. These are
  software-contract tests, not model accuracy measurements.
- Read-only audit: outputs/integrated_protocol_audit_20260906_b.
  Row counts remain train3992, validation573, test569, dev50=50.
  Pixel-group overlaps: train/validation7, train/test5, validation/test2,
  dev50/train49. Four near-duplicate candidates await adjudication.
  Fresh train-only eligible partition: development3100, calibration791,
  excluding inspected dev50 and external validation/test image groups.
- Qualification A: interrupted cache-free judge path after several minutes;
  checkpoints and failure note retained.
- Qualification B: judge output failed its schema budget; not semantic abstention.
- Qualification C: one development case completed in221.17s. Judge JSON valid,
  326tokens, approximately6.93GB peak allocated. Proposed contradiction rejected
  by fresh image checks; correct baseline retained. Saturated clauses and Agent2
  modifier errors remained. This is not an accuracy gain or qualification of
  subsequent code.
- Qualification D completed all three cases, selected first-in-source-order per
  phenomenon without consulting labels/predictions. Wall time957.74s; baseline
  and final answers identical, 2/3 correct, no accepted correction. All three
  reviews failed clause-completeness after retry; all three caption witnesses
  were unusable. Runtime success is not semantic abstention. Actual generation
  requests25/23/26 (74total), known output tokens2853/2425/3050. Output:
  outputs/integrated_qualification_20260906_d.
- The independent-vote control restored the same upstream cases and saved all
  three predictions with zero new generation requests, but final timing export
  crashed due to a locally scoped deepcopy import. The import is fixed and a
  full cached-path regression passes. Original run retained as failed_finalization;
  separate post-failure metrics do not turn it into a successful experiment.
- Post-D repairs withhold unusable witness prose and invalid expected/opposite
  states from the judge. The text ceiling is320characters (soft target160),
  output reserve1024tokens, unchanged total context6144 and120s per-generation
  deadline. Focused review-stage replay E completed against D's saved hearings:
  all three still failed clause completeness after two attempts each. Maximum
  observed allocated memory7.0644GB; no OOM recovery. Increasing the allowance
  did not qualify the contract. This is not a new full-pipeline or held-out run.
- Internal development/calibration manifests were successfully generated and
  identity-checked:3100rows/2103groups and791rows/526groups. Separate provenance
  sidecars declare dataset_mutated=false. Blinded review packets for D exist at
  outputs/blinded_qualification_20260906_d; all ratings remain blank.
- Installed dependency, control-plane and pinned judge/vision file checks passed.
  Ten-source archive: all40file hashes verified, none mismatched.

## Remaining work — not complete

1. Resolve the failed judge/witness qualification before running a larger
   experiment. A larger text allowance alone did not work. Qualify the latest
   full pipeline and conventional control on identical frozen inputs; do not
   treat failed control finalization as successful qualification.
2. Agent2's valid schemas still contain semantic errors. Its same-family caption
   auditor can approve incomplete expected states and reject useful contrary
   states. Qualify extraction/auditing on fresh internal development groups before
   replacing or fine-tuning models.
3. Annotate statement-level bridge/alternative faithfulness. A fresh image-relation
   check does not prove every rationale clause. Heuristic family/prose gates
   remain and may cause false rejection.
4. Fit/validate calibration and select acceptance policy on disjoint development
   data, then freeze thresholds. Current heuristic gates were not relaxed.
5. Complete compute-matched voting/self-consistency, ablations, adequate held-out
   image-group-aware comparisons and prespecified stochastic seeds. Small dev
   runs cannot establish the prespecified five-point improvement target.
6. Obtain actual independent human ratings and measure claim/observation/citation/
   alternative/explanation correctness, false corroboration/rejection and agreement.
7. Expand real-output replay, controlled image counterfactuals, failure recovery
   and all-model cost/memory qualification. Generation accounting is not FLOPs.
8. Gated upstream rebuild/provenance verification awaits the user's later access;
   near duplicates need adjudication. Do not change established split membership.

The contribution under investigation remains figurative claim preservation,
targeted cross-modal disputes, fresh image verification and faithful revision
traces—not invention of debate or multimodal debate.
