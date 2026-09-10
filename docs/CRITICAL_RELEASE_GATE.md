# Critical repair release gate

These criteria apply before broader qualification outcomes are examined.
Previously inspected development cases remain diagnostic examples, not unseen
validation data. No acceptance threshold in the tribunal is relaxed.

1. All deterministic type, identity, source-span, dependency, citation, cache and
   logical-composition tests must pass. Historical expected failures may remain
   only in a clearly identified retired diagnostic, never a live path.
2. All required schema/tokenizer conformance cases must pass. Record unsupported
   features and property-order restrictions; do not claim exhaustive coverage.
3. Independently reviewed semantic fixtures must cover correct, incorrect and
   ambiguous representations across roles, negation, quantities, degree, scope,
   idioms, sarcasm and bound comparisons. Two human reviews and adjudication are
   required. The supplied packet is deliberately unannotated.
4. In that frozen finite suite, require each declared valid obligation to pass
   and each declared invalid obligation to fail. Unknown is acceptable only where
   the annotation permits it. Report false approval, false rejection, unknown,
   formatting failure and execution failure separately. This finite-suite gate
   is not an all-input guarantee or an accuracy claim.
5. A fresh full-pipeline development run must complete on pinned source/configuration
   with no hidden decoder fallback or stale checkpoints. A correct baseline answer
   with an invalid claim graph does not qualify the semantics.
6. Baseline and tribunal share repaired caption/decoder infrastructure. Demonstrate
   improvement using the existing paired image-group-aware evaluation and declared
   controls, not diagnostic cases or selected successful outputs.
7. Final explanations require independent human faithfulness review. Archive
   sources, adaptations, versions, negative results and costs.

Current release is not qualified: Mistral failed a controlled semantic case,
Agent 2 still produced missing graph conditions in the full run, and independent
annotations and held-out comparisons remain incomplete. Qwen's four passing
diagnostic variants do not waive this gate or establish general qualification.
