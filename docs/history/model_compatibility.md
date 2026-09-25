# Task-Trained Model Compatibility Decision

## Candidate

`asaakyan/LLaVA-1.5-7b-eViL-VFLUTE-lora` is the official checkpoint linked
from the V-FLUTE dataset and paper:

- https://huggingface.co/asaakyan/LLaVA-1.5-7b-eViL-VFLUTE-lora
- https://huggingface.co/datasets/ColumbiaNLP/V-FLUTE
- https://aclanthology.org/2025.naacl-long.1/

## Decision

Do not make this checkpoint the default FigDebate visual agent.

The published usage path depends on the original custom LLaVA repository and
loads `llava-v1.5-7b` plus the task adapter through that repository's builder.
FigDebate uses Hugging Face `LlavaForConditionalGeneration` with a pinned HF
conversion and 4-bit loading. The model card also describes the published
checkpoint as a 7B model with F32/F16 tensors and demonstrates `load_4bit=False`.
That path is not a verified drop-in replacement for the current loader and is
not appropriate as a default on the validated 8 GB RTX 4060 Laptop GPU.

It was trained directly on V-FLUTE (and e-ViL), so it is useful as an external
task-trained baseline. It must not be presented as FigDebate's novel component.
Making it the default before a matched compatibility benchmark would confound
the contribution from debate, evidence governance, and procedural memory.

## Acceptance Gate For A Later Baseline

The checkpoint may be added only as a separate `task_trained_baseline` after:

1. Its original base model and adapter load successfully in an isolated
   environment.
2. A 4-bit inference path fits the available GPU without changing outputs.
3. It emits a parseable binary label and explanation on the three-sample smoke
   set.
4. It completes the same stratified validation subset and seed as FigDebate.
5. Its model revision, loader commit, runtime, and memory use are recorded.

This task-trained LLaVA checkpoint remains an external baseline. The active
FigDebate Agent 1 is pinned Qwen3-VL 4B Instruct, loaded in NF4 4-bit mode and
used only for short visual questions. Python validates its answers and
deterministically assembles the evidence schema. Mistral remains responsible
for claim/decision reasoning, with deterministic evidence governance and the
existing selective debate controls.
