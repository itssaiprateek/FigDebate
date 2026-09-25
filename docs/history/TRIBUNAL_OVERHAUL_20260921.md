# Tribunal qualification, 21 September 2026

The semantic overhaul candidate was **not promoted**. Normal runs retain the
current judge, initial arbiter and gate rules, with feedback disabled.

Retained changes:

- Source interval decoding enforces `0 <= start < end <= token_count` while
  preserving the existing wire representation and exact source copying.
- All invalid source intervals can be repaired together within one retry;
  unchanged fields are retained and the entire result is validated.
- Duplicate visual selections can repair only the selection array, preserving
  the original coverage judgment. Semantic truth is not established by valid IDs.
- Gate metadata records the decision path, stage input hashes, source bindings,
  repairs, and distinct outcomes for agreement/correction/missing evidence/
  unresolved interpretation/execution failure. Inconsistent judgments are
  distinguished from malformed output.
- Capped verification clauses without sentence completion stop safely. Automatic
  shortening was tested but rejected because it could change the explanation's
  stance. Existing uncapped field repairs remain bounded; no retry loop was added.

Four arms replayed the same five selected development cases using frozen first
hearings and the real review/gate path. Baseline, repair-only, preventive decoder,
and focused-dialogue arms accepted **zero corrections and zero harms**. All kept
3/5 final labels correct. Preventive decoding removed the observed invalid-span
failure. One inconsistent judgment remained.

The research-only candidate added two independently generated interpretations,
one grounding objection and a bounded reply before the normal proposal. Harmful
proposals increased from 2 to 3, and review time increased from approximately
337 to 985 seconds. Its 48 calls produced no accuracy benefit. It is not imported
by production and has no normal-run enablement flag.

The final software suite has 260 checks: 259 pass and one pre-existing expected
failure concerns the unchanged initial arbiter. Two old mock-runtime fixtures
were corrected without changing assertions. Decoder probes and bounded repair
tests are recorded with the experiment.

No sample-specific production rules or human reference explanations were added.
These are targeted diagnostics, not a held-out accuracy estimate or proof of
stability across 6,000 examples. The tribunal's semantic accuracy problem remains
unresolved. Consolidation, gate calibration and alternative model families were
not silently rolled out after the role-quality experiment failed.

Full artifacts: `../../research/tribunal_overhaul_20260921/REPORT.md`,
`scored_results.json`, `final_audit.json`, `source_audit.json`, and the per-arm raw
requests/responses. The paired runs precede the final cap guard; that guard has
separate targeted checks. No full dataset run was performed.
