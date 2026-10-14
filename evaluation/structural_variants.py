"""Unpromoted 2026-10-02 ablations, enabled only by the evaluation entry point."""
# Fixed authored method demonstrations, not dataset examples or retrieved answers.
# Balanced supported/unsupported connections; supplied only in the evaluation arm.
SEMANTIC_EXAMPLES = (
    'Method examples (unrelated invented cases, never evidence for this case): '
    'Image: a worker juggling several clocks. Claim: she is juggling deadlines. '
    'SUPPORT can follow from the depicted clock/deadline analogy, not literal clock juggling. '
    'Same image. Claim: she met every deadline yesterday. UNRESOLVED: the analogy does not establish completion or yesterday. '
    'Image: a handmade trophy on an empty desk. Claim: the trophy is on the desk. SUPPORT from location. '
    'Same image. Claim: its owner won honestly. UNRESOLVED: possession does not establish fair conduct. '
    'Image: a character promises to help but is then shown walking away. Claim: the character helped. '
    'CONFLICT requires the depicted contrary outcome; the promise alone would leave the outcome unresolved. '
    'Always assess the actual supplied image and complete caption independently of these examples. '
)


def enable_evaluation_modes():
    from engine import semantic_protocol as semantics
    modes = {'canonical-audit-1': {'conditions','canonical'}, 'examples-audit-1': {'examples'}, 'paired-claim-audit-1': {'conditions','canonical'}, 'paired-defined-audit-1': {'conditions','canonical'}}
    semantics.EXPERIMENTS.update(modes)
    semantics.MODES.update(modes)
    semantics.SEMANTIC_EXAMPLES = SEMANTIC_EXAMPLES
