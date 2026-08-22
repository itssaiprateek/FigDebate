# FigDebate Code Audit

> Historical audit: this document records the pre-unification system and its
> original problems. It is retained for change history. `README.txt`,
> `run_figdebate.py`, and the latest run folder are authoritative for the
> current system.

## Current Status

The project is a working multimodal figurative-language entailment system. It currently contains development-phase naming (`phase4`, `run_phase4.py`) and multiple entry points that run similar but not identical versions of the pipeline.

The final research system should be presented as one unified framework:

```text
Image + Caption
      |
      v
FigDebate
      |
      v
Prediction: label, confidence, explanation, diagnostics
```

## Entry Points

### main.py

Purpose:

- Runs a single example from `capcon_test.pkl`.
- Loads the image and caption.
- Instantiates `FigDebatePipeline`.
- Prints the ground-truth label, predicted label, and confidence.

Assessment:

- Useful as a smoke-test/demo script.
- Not the official experiment runner.
- Calls `pipeline.py`.
- Does not call `run_phase4.py`.

### pipeline.py

Purpose:

- Defines `FigDebatePipeline`.
- Runs Agent 1, unloads LLaVA, loads Mistral, runs Agent 2, comparator, and Arbiter.
- Returns the major intermediate outputs and final decision.

Assessment:

- This is a single-sample pipeline.
- It currently imports `comparators.comparator.compare`, the older literal-overlap comparator.
- It does not use the Phase 4 debate engine or feedback loop.
- It does not call `phase4.orchestrator.Orchestrator`.

Important issue:

- The current Agent 1 and Agent 2 public outputs are spec-shaped:

```text
Agent 1: visual_description, objects, scene_type, symbolic_tone
Agent 2: surface_meaning, figurative_type, intended_meaning, background_knowledge
```

- The older comparator expects internal fields:

```text
Agent 1: people, objects, actions
Agent 2: key_entities, key_actions, non_literal_expressions
```

- Therefore `pipeline.py` can lose information unless it passes `_internal` fields into the old comparator or switches to the newer comparator.

### run_phase4.py

Purpose:

- Loads the dev split.
- Instantiates `phase4.orchestrator.Orchestrator`.
- Runs multiple samples.
- Writes `phase4_results.csv`.
- Prints basic accuracy and runtime statistics.

Assessment:

- This is the current Phase 4 evaluation runner.
- It runs the most complete system because it uses the orchestrator, debate engine, feedback loop, and `comparator_v2`.
- It should eventually be replaced by a phase-neutral runner such as `run_figdebate.py`.

Current issue:

- The script says `NUM_SAMPLES = 11`, while the existing `phase4_results.csv` contains 14 result rows. The result file likely came from an earlier run or a manually changed sample count.

### test_pipeline_dev.py

Purpose:

- Runs the full Step 3 dev-split evaluation with checkpoint/resume logic.
- Produces:
  - `step3_agent1_checkpoint.jsonl`
  - `step3_full_pipeline_results.jsonl`
  - `step3_dev_split_results.csv`
- Computes accuracy and macro F1.

Assessment:

- This is an experiment/evaluation script, not the final production entry point.
- It is still useful for reproducible Step 3 baseline comparison.
- It should not be deleted until a new phase-neutral evaluation runner fully replaces it.

## Core Modules

### agents/agent1.py

Role:

- Visual grounding agent.
- Uses LLaVA.
- Receives only the image.
- Produces visual evidence.

Public output:

```text
visual_description
objects
scene_type
symbolic_tone
_internal
```

Important notes:

- The raw prompt asks for more detail than the public schema exposes.
- Useful internal fields such as `visual_facts`, `possible_visual_metaphors`, and uncertainty are kept only under `_internal`.
- Some prompt text appears mojibake-encoded in the file (`â€¢`, `â€“`, `â†’`, `âœ“`), probably from encoding conversion.

### agents/agent2.py

Role:

- Caption and figurative-language understanding agent.
- Uses Mistral.
- Receives only the caption.
- Produces caption interpretation.

Public output:

```text
surface_meaning
figurative_type
intended_meaning
background_knowledge
_figurative_type_was_guessed
_internal
```

Important notes:

- The prompt currently tells the model to prefer `metaphor` when uncertain.
- The fallback parser also returns `metaphor` when no valid type is found.
- This explains the observed metaphor bias in Step 3 results.

### comparators/comparator.py

Role:

- Older literal keyword overlap comparator.

Assessment:

- Useful for historical Step 3 baseline.
- Not ideal as the final comparator for figurative-language entailment.
- It now warns that literal overlap is not an entailment score.

### comparators/comparator_v2.py

Role:

- Theme-based comparator used by Phase 4.

Assessment:

- More compatible with the current Agent 1 and Agent 2 public schemas.
- Still has logic that can treat weak/no semantic overlap as conflict.

Important issue:

- Missing support and contradiction should be separated:

```text
supporting_points
possible_conflicts
neutral_notes
```

Only specific observable incompatibilities should be treated as conflict.

### arbiter/arbiter.py

Role:

- Final binary decision maker.
- Uses Mistral.
- Receives caption, visual grounding, language understanding, and comparator output.

Output:

```text
label
explanation
confidence
debate_needed
_label_was_forced
_internal
```

Important notes:

- The prompt correctly says absence of support is not contradiction.
- The parser/fallback logic is still vulnerable when the model outputs `Final Decision: None`.
- Forced labels are tracked and automatically trigger debate.

### phase4/orchestrator.py

Role:

- Current most complete orchestration layer.
- Runs Agent 1, Agent 2, Comparator V2, Arbiter, optional Debate, and Feedback.

Assessment:

- This is currently the best internal engine.
- It should be wrapped by a phase-neutral public API rather than renamed immediately.

### phase4/debate.py

Role:

- Runs Agent 1 critique and Agent 2 critique when confidence is low or the label was forced.
- Sends critiques back to the Arbiter for revision.

Assessment:

- Debate is currently triggered frequently because confidence is often low or labels are forced.

### phase4/feedback_loop.py

Role:

- Stores examples of common failure types.
- Builds feedback prompts for future Agent 1 and Agent 2 runs.

Assessment:

- Useful conceptually, but current feedback memory is only in-process and updates every 50 predictions.

## Dataset Structure

Processed files:

```text
dataset/data/processed/vflute_train.pkl
dataset/data/processed/vflute_val.pkl
dataset/data/processed/vflute_test.pkl
dataset/data/processed/mmsd2_train.pkl
dataset/data/processed/mmsd2_val.pkl
dataset/data/processed/mmsd2_test.pkl
dataset/data/processed/capcon_train.pkl
dataset/data/processed/capcon_val.pkl
dataset/data/processed/capcon_test.pkl
dataset/data/processed/dev_split.pkl
```

Each sample contains:

```text
id
caption
explanation
image_bytes
label
phenomenon
source
```

Dev split:

```text
50 samples
25 ENTAILS
25 CONTRADICTS
13 metaphor
19 sarcasm
18 humor
```

## Existing Results

### Step 3

File:

```text
step3_dev_split_results.csv
```

Observed metrics:

```text
50 samples
28 correct
56.0% accuracy
Predicted ENTAILS: 23
Predicted CONTRADICTS: 27
Figurative type predictions:
- metaphor: 43
- sarcasm: 3
- humor: 4
```

Main issue:

- Agent 2 is strongly biased toward metaphor.

### Phase 4

File:

```text
phase4_results.csv
```

Observed status:

```text
14 rows
8 correct
57.1% accuracy
Predicted ENTAILS: 1
Predicted CONTRADICTS: 13
```

Main issue:

- Phase 4 currently appears heavily biased toward CONTRADICTS.

## Execution Flow

### Single-sample demo path

```text
main.py
  |
  v
pipeline.FigDebatePipeline
  |
  v
LlavaModel -> VisualGroundingAgent
  |
  v
MistralModel -> ClaimExtractionAgent
  |
  v
comparators.comparator
  |
  v
Arbiter
  |
  v
Result
```

### Step 3 evaluation path

```text
test_pipeline_dev.py
  |
  v
load_dev_split
  |
  v
LlavaModel -> VisualGroundingAgent
  |
  v
step3_agent1_checkpoint.jsonl
  |
  v
MistralModel -> ClaimExtractionAgent
  |
  v
comparators.comparator using _internal outputs
  |
  v
Arbiter
  |
  v
step3_dev_split_results.csv
```

### Phase 4 path

```text
run_phase4.py
  |
  v
phase4.orchestrator.Orchestrator
  |
  v
LlavaModel -> VisualGroundingAgent
  |
  v
MistralModel -> ClaimExtractionAgent
  |
  v
comparators.comparator_v2
  |
  v
Arbiter
  |
  v
DebateEngine if needed
  |
  v
FeedbackLoop
  |
  v
phase4_results.csv
```

## Essential Files

```text
agents/agent1.py
agents/agent2.py
arbiter/arbiter.py
comparators/comparator_v2.py
dataset/loaders.py
models/model_loader.py
models/mistral_loader.py
phase4/orchestrator.py
phase4/debate.py
phase4/feedback_loop.py
phase4/gpu_manager.py
utils/parser.py
utils/claim_parser.py
utils/arbiter_parser.py
```

## Historical Or Evaluation Files

```text
pipeline.py
test_pipeline_dev.py
run_phase4.py
comparators/comparator.py
phase4_results.csv
step3_dev_split_results.csv
step3_agent1_checkpoint.jsonl
```

These should not be deleted yet because they preserve baselines and reproducibility.

## Files That Appear Obsolete Or Redundant

```text
utils/prompts.py
utils/alignment_parser.py
dataset/data/loaders.py
```

Notes:

- `utils/prompts.py` contains older prompt templates, while the active agents define prompts internally.
- `utils/alignment_parser.py` is empty.
- `dataset/data/loaders.py` duplicates `dataset/loaders.py`.

## Step 2 Target

The next structure should introduce a phase-neutral public API without deleting historical files:

```text
figdebate.py
run_figdebate.py
```

The public API should expose:

```python
from figdebate import FigDebate

system = FigDebate()
result = system.predict(image, caption)
```

Internally, this can wrap the current `phase4.orchestrator.Orchestrator` first. Later, after experiments are stable, the `phase4` package can be renamed to `engine`.
