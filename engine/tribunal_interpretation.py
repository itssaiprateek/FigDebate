"""Common task semantics for the proposer and separately prompted verifier."""

TASK_SEMANTICS_VERSION = 'vflute-source-assertion-2'
# The locked loader preserves the upstream ENTAILMENT/CONTRADICTION labels.
# These are inference obligations, not a new dataset relabeling convention.

INTERPRETATION_RULES = (
    "Distinguish factual assertions from author evaluations, quoted character speech and figurative mappings. "
    "An evaluative reaction can be grounded by the depicted trigger and expressed stance; it does not require "
    "the evaluative adjective to be printed in the image or proof of a person's hidden intent. "
    "A reaction meme may analogize a visible emotional response to a caption situation without depicting its "
    "literal historical participants. Explain the correspondence and preserve necessary roles and qualifiers. "
    "Do not turn this into automatic support: unrelated reactions, opposite attitudes, mismatched roles and "
    "unsupported factual claims still require scrutiny. Distinguish an offer or quoted message from a completed "
    "action and its actual outcome. Read complete relevant image text with its speaker/panel attachment before "
    "interpreting a fragment. Resolve a conventional idiom in context before demanding literal events. "
    "Missing evidence is UNRESOLVED, never by itself CONFLICT. Do not reverse sarcasm merely because it is sarcasm. "
)

V5_INTERPRETATION_RULES = INTERPRETATION_RULES + (
    "Judge the complete source assertion under its context-licensed reading. Preserve negation, qualifiers, "
    "subject, property and scope. Speaker attitude is distinct from the assertion; an observer's reaction "
    "is distinct from a depicted participant's property or outcome. In paired comparisons keep each actual outcome "
    "attached to its own subject. Apply the same standard to SUPPORT and CONFLICT. "
)
