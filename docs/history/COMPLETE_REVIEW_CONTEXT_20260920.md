# Tribunal context delivery

The previous retrieval implementation preserved already retrieved records, but
still allowed the proposer to request new context after its last retrieval cycle.
The latest dev10 showed four final reviews blocked by this condition.

For evidence-review-5.0, all available candidate arguments and caption readings
are now disclosed before the first proposal. Their IDs are protected when fitting
later retrieval prompts. They retain their status as fallible context, not facts
or independent verification. No human reference explanations or gold labels are
introduced into the prompt.

The Qwen runtime measures the actual image and chat-wrapper token overhead on CPU
and assigns the remaining total allowance to text, reserving the complete output
allowance and 64 tokens for tokenization boundary differences. The existing total
token limit and exact combined-length check remain in force. The source caption,
claim graph and current hearing are never truncated. A context packet that cannot
fit fails explicitly; this does not guarantee every arbitrarily large case fits.

Image-observation records beyond the detailed ledger still use the existing
retrieval mechanism. This change guarantees delivery of the available argument
and reading catalogue when budgeting succeeds; it does not imply that every
observation is in the initial prompt, that missing real-world facts are known,
or that a model interprets disclosed information correctly.

The initial arbiter, evidence acceptance requirements, configured production
judge and disabled-feedback policy are unchanged. Existing runs must not be
resumed across this source change.

Validation: 136 focused software tests passed (56 context/repair checks and 80
proof, recovery and completion checks). Offline replay of all ten archived
first-hearing inputs disclosed every context record under the unchanged 6,144
total token limit. A further CPU-only audit passed all 15 archived first and
second hearings, with no undisclosed argument/reading records. Live 4B/9B
diagnostics are recorded separately under
`research/judge9b_context_20260920`; development results cannot establish a
dataset-wide accuracy gain.
