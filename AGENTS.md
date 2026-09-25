# User standing instructions

- Improve mechanisms for the dataset and the target figurative-language tasks (humor, metaphor, sarcasm). Never add production rules keyed to individual sample IDs, image hashes, captions, gold labels, or expected answers. Identity checks for provenance/leakage prevention and named regression fixtures are allowed.
- Keep the initial arbiter unchanged unless the user explicitly changes that scope. Current improvement work concerns the tribunal and feedback.
- Use bounded targeted checks that exercise the real review and acceptance paths. Leave full dataset runs to the user unless they explicitly request one.
- Separate execution reliability, correction quality, harmful changes, feedback contribution and runtime. Selected development regressions cannot establish held-out accuracy or a guaranteed gain above ten percentage points.
- Current evaluation focus is the tribunal alone. Keep feedback disabled in all future suggested commands and runs (`--feedback-mode disabled`) until the user explicitly asks to re-enable it. Do not reuse the older integrated-feedback command or resume an integrated-feedback run for a tribunal-only comparison.
