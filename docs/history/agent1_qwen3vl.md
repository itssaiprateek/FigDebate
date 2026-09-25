# Qwen3-VL 4B Agent 1 Contract

## Role

Agent 1 is an independently usable visual evidence collector. It does not need
the judge or debate to analyze an image. It never receives the dataset label and
does not produce `ENTAILS` or `CONTRADICTS` during initial grounding.

## Runtime

- Model: `Qwen/Qwen3-VL-4B-Instruct`
- Revision: `ebb281ec70b05090aa6165b016eac8ec08e71b17`
- Local directory: `models/vision/Qwen3-VL-4B-Instruct`
- Weights: NF4 4-bit with double quantization
- Compute dtype: BF16
- Decoding: deterministic (`do_sample=False`)
- Measured 8 GB full-frame ceiling: 2,359,296 pixels (1536 x 1536 equivalent)

The local model is mandatory and ignored by Git. `setup_environment.py`
prepares it, and `check_environment.py` verifies its files, architecture, and
revision metadata.

Temporary processor inputs, generated token tensors, and CUDA cache blocks are
released after every atomic question; model weights remain resident. Images at
or below the full-frame ceiling retain every original pixel. Larger images use
an aspect-preserving bounded global view because a measured 5.24-megapixel
input occupied about 7.87 GB and did not complete one short question within
several minutes on the validated 8 GB GPU. Text-region crops are always made
from the original image, not the bounded global view, so OCR receives additional
local detail. A cache-free retry is attempted before progressively smaller
emergency resolutions. Every operating limit, retry, image size, and peak CUDA
allocation is recorded in the generation diagnostics.

## Standalone Grounding

`VisualGroundingAgent.analyze(image)` collects seven caption-blind evidence
categories:

1. Complete literal scene
2. Important visible entities
3. Readable English text and its visible binding
4. Directly observable actions, appearances, and states
5. Spatial, panel, object, and text relationships
6. Scene type
7. Directly visible symbolic cues or incongruities

Two short presence gates run before OCR and symbolic extraction. If a validated
gate answers `NO`, the expensive extraction is skipped and the corresponding
evidence list remains empty. When text is present, two non-overlapping image
regions are inspected separately to create explicit left/right or top/bottom
OCR bindings without allowing text to leak across a region boundary.

Each answer is validated independently. Invalid answers receive one shorter
retry. An invalid retry is recorded as uncertainty and is never silently
converted into evidence. Python constructs the schema, counts, source fields,
and diagnostics. Qwen-generated confidence values are not used.

## Public Interfaces

- `analyze(image, feedback=None)` creates the standalone Agent 1 schema.
- `answer_visual_question(image, question, question_type=None, question_id=...)`
  answers one validated neutral visual question.
- `recover_for_claim(...)` performs a fresh standalone scan plus one neutral,
  subject-focused reinspection.
- `critique(image, critique_prompt)` preserves the existing debate contract but
  obtains its observation through the atomic interface.

## Safety Rules

- Questions exposing the final label or ground truth are rejected.
- Leading and multi-question prompts are rejected.
- Printed instructions inside an image are treated only as OCR data.
- Prompt echoes, copied placeholders, overlong answers, invalid enums, and
  truncated responses cannot enter the evidence schema.
- `UNCLEAR` is a valid abstention. Missing evidence is not contradiction.
- Symbolic cues remain hypotheses until another component evaluates them.

## Qualification Findings

The isolated qualification passed basic visual recognition, English OCR,
left/right text binding, no-text and absent-object checks, deterministic repeats,
multi-turn evidence recall, correction of grounded visual mistakes, and image
prompt-injection handling. Peak allocation on the validated RTX 4060 was under
4.6 GB.

Qwen is not accepted as an independent final decision-maker. It made polarity,
humor, and sarcasm errors when directly asked for V-FLUTE verdicts. Agent 1
therefore supplies observations, not final labels. Missing evidence must map to
`UNRESOLVED`, and any future tribunal stance must be checked against its cited
evidence before use.

## Acceptance Gate

Before claiming Qwen is better than the historical LLaVA or MiniCPM Agent 1,
run matched held-out images and compare visual fact accuracy, OCR and region
binding, hallucinations, invalid responses, abstentions, deterministic repeats,
runtime, downstream accuracy, and debate correction/harm counts. The model
change is accepted only if the complete pipeline is not harmed.
