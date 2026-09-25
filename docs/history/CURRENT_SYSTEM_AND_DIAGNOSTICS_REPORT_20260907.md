# FigDebate: current system, observed problems and diagnostic findings

**Report date:** 7 September 2026  
**Purpose:** explain the current project in plain language, document what happened in the latest ten-case run, and distinguish demonstrated defects from open research questions.  
**Scope:** current local implementation and saved experiments—not a claim that all planned repairs are complete.

## 1. Executive summary

FigDebate is a research system for deciding whether an image supports or contradicts a caption involving humor, metaphor or sarcasm. It is intended to do more than produce a label: it should preserve the caption's meaning, examine disputed evidence, consider alternatives, and leave an understandable record of why its final answer was chosen.

The basic idea is a small tribunal. One agent examines the image, another analyzes the caption, an initial decision-maker combines their work, and a supervisor reviews difficult cases. Software rules decide whether the supervisor's proposed correction is sufficiently supported to replace the initial answer.

The project is not a simple majority vote. The intended advantage is better evidence and reasoning—not merely more opinions.

The latest examined run completed ten cases. The initial system answered six correctly, and the final system still answered the same six correctly. Every final label was unchanged. Nine supervisor responses failed validation; the remaining response proposed a gold-matching correction but did not pass independent evidence verification. The tribunal therefore added substantial computation without changing an answer in this run.

The subsequent diagnosis found a particularly important engineering defect: the constrained decoder can block a legal token that closes a short sentence and its JSON field. This creates a concrete route to the unfinished explanations seen in the judge's output. Separately, controlled live tests showed that the caption auditor did not distinguish a correct supporting condition from missing or reversed support. Deterministic tests also exposed placeholder acceptance and loss of list types between stages.

These are substantive findings. However, fixing the decoder alone will not prove semantic correctness, and ten debugging cases cannot establish overall research performance.

### Current position in one table

| Question | Evidence-backed answer |
|---|---|
| Can the program complete a run and save results? | Yes: ten of ten cases completed in the examined run. |
| Is the current tribunal improving the answers in that run? | No: initial and final accuracy were both 60%, with identical labels. |
| Are most caption requirements reliable? | No: only one of ten hearing requirement sets passed the current validator, and even that set contains suspicious directional content. |
| Is the judge reliably returning usable decisions? | No: only one of ten responses passed its contract. |
| Is this primarily genuine judge uncertainty? | No: the valid semantic abstention count was zero; invalid outputs must be distinguished from abstention. |
| Have concrete causes been found? | Yes: a decoder token-filtering defect, a semantic-auditor discrimination failure, and deterministic boundary defects. |
| Are these causes repaired? | No: the latest diagnostic pass intentionally changed no production behaviour. |
| Is the three-category dataset selection itself the problem? | No such finding. Humor, metaphor and sarcasm remain the intended scope. |
| Is the system ready for publication claims of superiority? | No: qualification, calibration, controls, held-out evaluation and human explanation review remain incomplete. |

## 2. What problem the project is trying to solve

Each case contains an image and a caption. The system must return one of two task labels:

- **ENTAILS:** the image supports the caption at the appropriate literal or figurative level.
- **CONTRADICTS:** the image conflicts with the caption at that level.

For example, a caption about an economy recovering may be illustrated with symbols rather than a literal economy. A phrase such as “heart sank” usually describes an emotional state rather than a physical movement. A sarcastic caption may state something positive while its intended criticism is negative.

These cases require careful separation of three things:

1. What the caption actually states.
2. What figurative interpretation could connect its words to the image.
3. Whether the image supports or contradicts the stated claim under a justified interpretation.

Replacing a sarcastic statement with its opposite can change the task being evaluated. Conversely, demanding literal physical evidence for an idiom can make a valid figurative relationship impossible to recognize. The system must avoid both errors.

The external task is binary, but intermediate components can report **UNRESOLVED**, **INSUFFICIENT** or **ABSTAIN**. These internal states mean a component cannot establish a direction. They must not automatically become CONTRADICTS.

### Research goal and intended contribution

The target is an empirically useful combination of:

- Preserving figurative claim meaning and its important qualifiers.
- Asking focused questions about actual evidence disputes.
- Checking proposed changes against the current image.
- Tracking the origin and validity of evidence.
- Producing faithful explanations of accepted changes and fallbacks.

The project should not claim to have invented multi-agent debate or multimodal debate. Its proposed contribution is this particular combination and its demonstrated value on the selected task. That value remains to be established experimentally.

## 3. Plain-language glossary

| Term | Meaning in this project |
|---|---|
| Claim | The proposition expressed by the caption. |
| Claim frame | A structured description of who or what the caption concerns, what it asserts, and its qualifiers. |
| Contract | Rules describing what a component must return and what makes that return usable. This is a software contract, not a legal contract. |
| Schema | The required names and types of fields in a structured response. |
| OCR | Reading text printed in an image. |
| Witness | An agent answering a focused question about either the image or caption. |
| Evidence ledger | A record of observations and derived claims, each with an identifier and history. |
| Directional evidence | Evidence that establishes support or conflict, not merely that a relevant object exists. |
| Semantic bridge | The explicit explanation linking an image observation to the caption's meaning. |
| Corroboration | Additional checking of a proposed observation or relationship. It is not automatically independent just because another call is made. |
| Gold label | The dataset answer used for evaluation, not supplied to the reasoning agents. |
| Calibration | Checking and adjusting whether reported confidence corresponds to actual correctness rates. |
| Ablation | A controlled experiment that removes or changes one component to measure its effect. |

## 4. Structure of the codebase

The active codebase is the local `FigDebate_main` repository. The experiment runner is `run_figdebate.py`; `run_reproducible.py` provides a reproducible launch workflow. The project contains several research modes, but this report focuses on the tribunal configuration used in the examined run.

| Area | Main responsibility |
|---|---|
| `dataset/` | Load examples, preserve identifiers and source fields, audit duplicates and splits, and prepare controlled selections. |
| `models/` | Load the pinned model weights and manage generation, quantization and image budgets. |
| `agents/` | Visual grounding, caption extraction and multimodal supervisor behaviours. |
| `comparators/` | Compare caption requirements with visual reports and identify missing or disputed relationships. |
| `arbiter/` | Produce the initial binary decision and its assessment. |
| `engine/` | Orchestrate stages, route hearings, maintain evidence, validate contracts, verify proposals and decide whether revisions are accepted. |
| `evaluation/` | Metrics, paired comparisons, research controls, calibration, human-review packets and diagnostic utilities. |
| `tests/` | Automated regression and contract checks. |
| `outputs/` | Saved run configurations, checkpoints, predictions, traces and evaluation artifacts. |
| `docs/` | Architecture notes, the change tracker, research protocol, attribution records and diagnostic reports. |

There are more agent roles than separately loaded model families. Reusing a model for a different role is economical, but a new role or fresh prompt does not guarantee independent errors.

## 5. Dataset setup and correct use

### 5.1 What is preserved

The current dataset scope is the existing humor, metaphor and sarcasm subset. The diagnostic work did not replace examples, change labels, rewrite images, or change stored split membership.

The saved protocol audit reports:

| Stored selection | Rows |
|---|---:|
| Training | 3,992 |
| Validation | 573 |
| Test | 569 |
| Inspected development selection, `dev50` | 50 |

The ten-case run used `vflute_train_dev50`, with seed 42 and stratified selection. It contained four humor, four metaphor and two sarcasm cases. It is a debugging selection from training data, not a clean final-test evaluation.

### 5.2 What the models should and should not receive

The image and raw caption belong to the inference task. Gold labels and reference explanations belong to evaluation. They must not influence the agents' answers or repair decisions.

The code has mechanisms for retaining an immutable raw caption, filtering judge dossier fields and checking input identities. Those mechanisms are important safeguards; their existence is not a proof that every possible future code path is leakage-free.

The dataset phenomenon label is also different from the caption's predicted figurative type. Humor or metaphor may be in the image, caption or their combination. Therefore, the reported 40% caption-type/phenomenon agreement is a diagnostic observation—not a valid statement that caption understanding is only 40% accurate.

### 5.3 Split overlap and research consequences

The audit found shared pixel-based image groups across selections:

| Selections compared | Shared image groups |
|---|---:|
| Train / validation | 7 |
| Train / test | 5 |
| Validation / test | 2 |
| `dev50` / train | 49 |

These are image-group counts, not automatically counts of identical image-caption-label rows. Related captions can share an image, so image-level grouping matters when claiming independent evaluation cases.

Separate internal manifests were created from eligible existing training records:

- Development: 3,100 rows across 2,103 image groups.
- Calibration: 791 rows across 526 image groups.

Those selections exclude the inspected `dev50` groups and external validation/test image groups. They do not change the underlying dataset. Four near-duplicate candidates still need adjudication, and clean upstream reconstruction remains deferred until the user arranges gated Hugging Face access.

**Conclusion:** keep the three categories. The necessary work concerns controlled use, provenance and transparent reporting—not changing the chosen research scope.

## 6. Model roles and memory organisation

| Component | Current model or mechanism | What it works with |
|---|---|---|
| Agent 1: visual witness | Qwen3-VL-4B-Instruct | The image; initially caption-blind questions. |
| Agent 2: caption analyst | Mistral-7B-Instruct-v0.2 | The caption and caption-derived analysis. |
| Caption semantic auditor | Another Mistral call | Raw caption plus structured extraction or requirements. |
| Initial Arbiter | Mistral-backed reasoning/scoring | Caption, visual reports, language analysis and comparison. |
| Candidate arguments | Qwen visual candidate and Mistral text candidate | Image/caption or caption plus observations, without initial peer exposure. |
| Supervisor / judge | Qwen3.5-4B | Image and a filtered case dossier. |
| Independent image checking | Fresh calls through the configured multimodal runtime | Current image, caption and proposed premises, without the previous verdict. |
| Text NLI checker | Existing small CPU NLI model | Textual relationships used diagnostically, not as visual proof. |
| Final revision gate | Deterministic Python rules | Contracts, citations, verified directions and evidence history. |

The large models use quantized weights to reduce memory use. Stagewise execution loads a model for its batch of work, then releases it before another large model's stage. This avoids requiring all models to fit in GPU memory simultaneously.

The examined run used an RTX 4060 Laptop GPU and the `8gb` profile. Its saved judge settings included a 1,024-token output allowance, 6,144 total input-plus-output budget, 120-second per-generation deadline, and approximately one-million-pixel judge image budget. The current runtime starts with caching enabled and has an out-of-memory fallback path.

The `paper-8gb` profile is intended to keep evaluation inputs consistent across machines; unlike the ordinary `8gb` profile, it has no smaller-image OOM retry list. Changing profiles can change the visible image budget, so profile choices must be controlled in research comparisons.

## 7. End-to-end working of the current pipeline

```text
Fixed case: image + original caption
        |
        +--> Agent 1: visual observations and image text
        |
        +--> Agent 2: expressed claim + interpretation hypothesis
                       |
                       +--> caption semantic audit / bounded repair
        |
        v
Evidence comparison + ledger + initial Arbiter answer
        |
Independent candidate arguments and uncertainty routing
        |
        v
Focused hearing, when needed
  image witness answers a visual question
  caption witness states support/conflict requirements
        |
        v
Filtered case dossier + image -> supervisor review
        |
        +--> output validation / one format-repair attempt
        +--> bounded evidence retrieval or follow-up when applicable
        +--> fresh image-bound verification of a usable proposal
        |
        v
Deterministic revision gate
        |
        +--> accept justified revision
        OR
        +--> preserve initial answer and record why revision failed
        |
        v
Final label + confidence + explanation/evidence trace + evaluation files
```

The following subsections explain what each stage is supposed to contribute and where its limits lie.

### 7.1 Loading and locking a case

The runner selects examples according to a manifest or selection configuration. It records identifiers, seeds, model revisions and source/configuration checksums. Image and caption hashes help detect accidental mismatch or reuse of evidence from another case.

Checkpoints preserve completed stages. Compatible upstream work can be reused, but downstream tribunal decisions must not be silently reused across different experimental conditions. A successfully restored checkpoint means the previous computation was reused; it does not mean a fresh model call occurred.

### 7.2 Agent 1: observing the image

Agent 1 is initially asked separate short questions about the scene, objects, visible text, direct facts, relationships, scene type and symbolic cues. Software assembles these answers into a common evidence structure. This is more controlled than asking the model to produce an unrestricted essay as the entire visual record.

OCR crops can help with printed text. In the Flash-bike case, the log shows a full-frame resize while OCR crops remain sourced from the original image. This is input-budget management, not automatically a failure.

Nevertheless, a statement tagged OBSERVED is still a model report. It is not equivalent to independently established truth. A model can read the right words but attach them to the wrong object, infer an unsupported emotion, or reverse a relationship.

### 7.3 Agent 2: extracting the caption claim

The core extraction records the proposition, subject, predicate, objects or targets, asserted property, relation family, support state, opposite state, negation, quantities, modifiers, and scope.

For “The economy recovered relatively easily,” both recovery and relative ease matter. A representation that records only recovery has not preserved the full claim.

A separate interpretation stage proposes a figurative type, literal meaning, intended meaning, alternative and polarity. The original caption remains the reference. These interpretation fields are hypotheses, not permission to rewrite the claim.

### 7.4 Caption semantic audit and repair

A fresh caption-only call returns six checks:

1. Is the expressed claim preserved?
2. Are entity roles preserved?
3. Are negation and quantities preserved?
4. Are modifiers and scope preserved?
5. Does the expected state support the whole claim?
6. Does the opposite state conflict with the whole claim?

Failed checks identify fields for bounded repair. The code also maps false checks to repair targets when the model forgets to list them.

This is intended to stop malformed reasoning before it reaches the tribunal. Its current weakness is fundamental: an auditor's Boolean answer can be wrong, and the contract treats those answers as important semantic authority.

### 7.5 Comparator and evidence ledger

The comparator asks whether observations actually address the caption's requirements. Merely mentioning the same theme or entity should not establish support.

For example, seeing money and an upward arrow may relate to economic improvement, but does not necessarily establish that recovery was easy. Seeing a heart does not itself prove that it is rotten.

The ledger gives observations and derived relationships identifiers. It records where they came from, whether they remain active, and which earlier evidence supports a later assertion. Superseded or unsupported evidence should not continue justifying a decision. Shared roots should not be counted twice as independent corroboration.

Generic text NLI can suggest relationships worth investigating. It is prohibited from turning textual similarity alone into verified image evidence.

### 7.6 Initial Arbiter decision

The Arbiter combines caption analysis and visual reports. In the examined run, all ten initial decisions used `position_balanced_semantic` scoring.

In plain language, the model scores support and conflict with the choices presented in both orders. The scores are combined to reduce a tendency to prefer the first-listed option. The higher score determines the binary answer.

This reduces one possible presentation effect. It does not make the reasoning correct, nor does it turn the score into a calibrated probability. In the products example, the wrong ENTAILS answer received a raw score of about 0.912 despite the relevant OCR being available.

### 7.7 Independent candidates and hearing selection

The visual and text candidate roles form short arguments without initial peer exposure. They describe a reading, observation, alternative and decisive question. Their claims are hypotheses, not new evidence merely because they are expressed by another role.

The router considers uncertainty, missing grounding and specific disputes. The intended outcome is a focused question such as “Which outcome belongs to each bottle?” rather than a generic request to argue harder.

All ten cases in the examined run were routed to a Level 2 hearing. That is an observed routing outcome; it does not by itself prove the router is defective.

### 7.8 The focused witness hearing

Agent 1 re-examines the image for a specific observation. Agent 2 states what would support and what would conflict with the caption. Their previous final label is not supposed to dictate these answers.

The caption witness also indicates whether the extraction is acceptable. ENDORSE in this role concerns the supplied analysis; it must not be confused with an image-caption ENTAILS label.

The key requirement is that support and conflict describe meaningful, properly bound conditions. “None,” a copied instruction, a generic topic word or an unrelated experiment is not such a condition.

### 7.9 The judge's case dossier

The dossier packages the raw claim, usable analyses, evidence, witness responses and relevant history. It filters out gold answers and the current answer that could anchor the judge. Invalid witness prose and unusable generated states are withheld by current filtering logic.

Context is finite. The full record remains stored, while the visible packet prioritizes evidence and may index lower-priority items. A bounded retrieval/re-review mechanism exists for omitted material. A judge must not cite an unseen entry as though it had read the full evidence.

### 7.10 Supervisor review and semantic bridge

The judge examines the image and dossier and returns structured fields: its best judgment, support/conflict relation, visual premise, caption premise, semantic bridge, counter-interpretation, citations, confidence and any requested follow-up.

A good bridge should be understandable without trusting the judge's authority. For the products case, it should explicitly bind disliked to months and loved to a week, then compare those bindings with the caption's opposite assertion.

The judge gets one format-repair retry when its output is unusable. This is distinct from a second substantive tribunal round. Current code allows up to two substantive rounds, with a further hearing/review only under bounded follow-up conditions and new information. None occurred in the examined ten-case run.

### 7.11 Independent verification and final gate

A usable proposal can undergo fresh image checks. The verifier receives the actual image, raw caption, proposed premises and observation IDs—not the previous answer, confidence or proposed bridge conclusion. The choices are presented in two genuinely reversed orders.

Image and proposal hashes bind the verification record to the case. Checks examine the visual premise, caption preservation, entity/scope consistency and relation agreement.

This is stronger than asking the judge whether it agrees with itself, but it still uses a model and does not establish independent error distributions. Nor does checking a premise prove every sentence in the final explanation.

The deterministic gate checks contracts, citations and evidence direction before allowing a revision. The base tribunal confidence threshold is 0.75, with additional bridge and evidence checks elsewhere. These thresholds are not yet a calibrated acceptance policy.

Rejection preserves the baseline answer. This is useful containment, but preserves baseline errors as well. A system that never accepts a revision can avoid revision harms while also providing no corrective benefit.

### 7.12 Final artifacts and evaluation

The output records the final label, confidence, initial answer, proposed changes, acceptance/rejection reasons, citations and evidence ancestry. The intended explanation trail makes the decision auditable.

Important saved files include:

| File | Purpose |
|---|---|
| `run_config.json` | Conditions, models, seeds, budgets and source identity. |
| `sample_manifest.json` | The selected examples and their identities. |
| `records.jsonl` | Detailed case records and nested traces. |
| `predictions.csv` | Tabular answers, stage statuses and diagnostics. |
| `stage_checkpoints/` | Intermediate outputs for controlled resumption. |
| `metrics_summary.txt` / `metrics.json` | Aggregate evaluation results. |
| `judge_analysis.csv` | Judge contracts, proposals and acceptance information. |
| `debate_log.jsonl` | Detailed hearing/debate records. |
| `run_timing.json` | Total time and loading information. |
| Explanation and provenance analyses | Citation, evidence and explanation diagnostics. |

A trace is a recorded justification, not proof that the justification is correct. Faithfulness requires checking that the stated reasons actually support the delivered answer.

## 8. What happened in the ten-case run

Run: `outputs/checkpointed_supervisor_v5_oomfix_10`, recorded on 7 September 2026. Configuration: stagewise execution, tribunal judge, corroborated semantic bridges, independent candidates, feedback disabled, no control condition.

| Measure | Result | Interpretation |
|---|---:|---|
| Completed cases | 10/10 | Execution and export completed. |
| Initial accuracy | 60% | Six initial answers matched gold. |
| Final accuracy | 60% | Same six answers remained correct. |
| Macro F1 | 0.5238 | Class-balanced F1 was weaker than headline accuracy suggests. |
| Predictions | 9 ENTAILS, 1 CONTRADICTS | Strong local skew toward support. |
| Gold class counts | 5 / 5 | The observed skew is not explained by an imbalanced ten-case label count. |
| ENTAILS recall | 100% | All five entailment cases were identified. |
| CONTRADICTS recall | 20% | Four of five contradiction cases were missed. |
| Valid Agent 2 claim contracts | 2/10 | Most structured relations were unusable under current checks. |
| Claim repair attempts | 9 | Only one was recorded successful. |
| Valid hearing requirements | 1/10 | The caption witness was a major bottleneck. |
| Valid judge contracts | 1/10 | Most supervisor proposals never reached meaningful acceptance assessment. |
| Judge format retries | 9 | Zero successful retries. |
| Accepted corrections / harms | 0 / 0 | No benefit; zero harms is unsurprising with zero changes. |
| Decision-grade directional evidence | 0 cases | Source citations did not establish verified direction. |
| Generation requests | 249 | Includes more than just the ten judge requests. |
| Wall-clock time | 2,222.34 seconds | About 37 minutes, or 3.7 minutes per case on average including overhead. |

Humor and metaphor each scored 2/4; sarcasm scored 2/2. Two sarcasm cases do not establish that sarcasm is solved or easier.

### The crucial reporting distinction

“Valid decisions: 10” means ten valid final binary outputs. It does **not** mean ten valid tribunal judgments, ten verified explanations, or ten independently grounded decisions.

The raw judge verdict distribution includes two ABSTAIN values, but both belong to invalid responses. The qualified semantic-abstention count is zero. The dominant problem here is unusable output, not responsible uncertainty.

## 9. Case-by-case results

In this table, “judge label” is a recorded proposed/raw value. Unless its contract is valid and its evidence accepted, it is not an eligible final correction.

| Case suffix | Topic | Gold | Initial → final | Judge label | Why no accepted revision |
|---|---|---|---|---|---|
| 3351 | Disliked/loved products | CONTRADICTS | ENTAILS → ENTAILS | ENTAILS | Incomplete semantic bridge. |
| 1540 | Black-hole analogy | ENTAILS | ENTAILS → ENTAILS | ENTAILS | Incomplete counter-interpretation. |
| 4536 | Rotten heart | CONTRADICTS | ENTAILS → ENTAILS | CONTRADICTS | Valid contract, but independent directional support failed. |
| 2760 | Economy recovering easily | ENTAILS | ENTAILS → ENTAILS | CONTRADICTS | Incomplete visual premise. |
| 1713 | Retirement sarcasm | CONTRADICTS | CONTRADICTS → CONTRADICTS | CONTRADICTS | Incomplete bridge/counter-interpretation. |
| 566 | Discriminatory dress code | ENTAILS | ENTAILS → ENTAILS | CONTRADICTS | Incomplete semantic bridge. |
| 2199 | Police speed question | CONTRADICTS | ENTAILS → ENTAILS | CONTRADICTS | Multiple incomplete fields. |
| 2850 | Flash's motorcycle | ENTAILS | ENTAILS → ENTAILS | ABSTAIN | Judgment–relation mismatch; raw fields also contain truncation. |
| 4315 | Calm faculty meeting | CONTRADICTS | ENTAILS → ENTAILS | CONTRADICTS | Multiple incomplete fields. |
| 1595 | “Heart sank” | ENTAILS | ENTAILS → ENTAILS | ABSTAIN | Incomplete visual premise/reason. |

Full IDs begin `vflute_train_`. The following examples draw on saved model reports, OCR and dataset reference material; this report does not claim a new independent human image annotation.

### 9.1 Products: correct text, wrong relationship

Caption: “Products one dislikes disappear quickly, while products one loves last a long time.”

The recorded image text places “LASTS FOR MONTHS” with the disliked product and “GONE IN A WEEK” with the loved product. That is the reverse of the caption's stated relationship.

Agent 1's hearing response describes those bindings, then says they support the caption. Agent 2 asks for “duration testing of products in real use” and supplies `none` as conflict. The Arbiter scores the wrong ENTAILS answer highly. The judge's bridge is unusable.

**Learning:** the missing capability is not merely reading more text. It is preserving which property belongs to which entity and comparing the two structures in the correct direction. Repetition of the correct OCR does not guarantee correct inference.

### 9.2 “Heart sank”: interpretation recognized, then lost

Agent 2 identifies disappointment or sadness in its interpretation output. Elsewhere it uses an expected state of `None`, proposes “heart floated up” as an opposite, and asks for `visual_evidence_of_heart_sinking` in the hearing.

The final ENTAILS answer matches gold, but the tribunal does not provide a valid explanation-backed contribution.

**Learning:** a correct final label can coexist with a faulty internal process. Figurative interpretation must be linked to requirements, not stored as a disconnected descriptive field.

### 9.3 Faculty meeting: audit rejection and copied instructions

Caption: “The faculty meeting was calm and relaxing.”

The saved analysis proposes “chaotic and tense” as an opposite. Its auditor marks the opposite check false, while giving a generic reason about fields being correctly preserved. The hearing support includes copied instruction text rather than an actual evidence condition. The judge proposes CONTRADICTS but its explanation fields are incomplete. The wrong baseline remains.

**Learning:** the failure exists both in semantic checking and in the translation from an analysis into a usable witness statement. This case also supplied the fixed caption for controlled diagnostic tests.

### 9.4 Rotten heart: a gold-matching proposal is not automatically a sound correction

Caption: “His heart within him is fully rotten.”

The only valid witness requirement set includes `evidence_of_fresh_heart` as support and `absence_of_evidence_of_rotten_heart` as conflict. The support direction is suspicious, and absence of evidence is not automatically affirmative contrary evidence.

The judge returns CONTRADICTS, matching gold. However, its reasoning relies on a healthy-looking anatomical heart while also acknowledging possible moral corruption. A healthy anatomical heart does not by itself logically exclude moral corruption.

Independent verification does not endorse the proposal; the gate rejects it. The final ENTAILS answer remains wrong.

**Learning:** this is evidence of an unrealized correction, but not proof that the gate should simply have accepted it. The proper objective is a correct answer with a justified bridge, not acceptance based on gold hindsight.

### 9.5 Economic recovery: dropping a qualifier changes the claim

Caption: “The economy recovered relatively easily.”

The hearing requirement refers to the absence of recession or contraction in the caption and supplies an empty conflict field. That neither establishes recovery from the image nor preserves the ease qualifier. The judge's premise ends unfinished while discussing manner or speed.

**Learning:** recovery, speed of recovery and ease of recovery are related but different claims. A repair that reduces all of them to positive growth can introduce systematic error.

### 9.6 Dress code: the wrong target of evaluation

The saved shirt text contrasts service rules for men and women, including “Women NO Shirt FREE Beer!” The caption criticizes the offer as misogynistic.

Agent 2 asks whether the women in the image are wearing shirts. That changes the evidence question from the content of an offer to the clothing of depicted people. The judge produces an invalid CONTRADICTS proposal, while the correct baseline is preserved.

**Learning:** normative language does not eliminate the need for precise factual grounding. The system must identify what is being judged before asking what evidence is necessary.

### 9.7 Police question and Flash's bike: requirements can become content-free

For the police question, Agent 2 supplies “quantities” as support and `none` as conflict. A category name is not an evidence condition. The judge's multiple unfinished fields prevent an otherwise gold-matching raw label from qualifying.

For Flash's bike, Agent 2 requests more context without producing a usable contrary condition. The judge has a judgment–relation mismatch. The correct baseline survives, but that does not validate the tribunal's reasoning.

**Learning:** uncertainty must be represented honestly, while malformed or non-responsive requirements must be treated as a distinct failure type.

## 10. Targeted diagnostics: what was tested and what was learned

### 10.1 Decoder test: a valid output path is blocked

The diagnostic used both actual installed tokenizers and the production tribunal schema. A short valid review fails at a token containing a period, closing quote and comma—the natural end of a sentence and transition to the next JSON field. The affected token IDs were 9191 for Mistral and 10152 for the Qwen judge tokenizer.

The underlying character grammar accepts the complete same text. Therefore the rejection occurs in token filtering, not because the JSON response is invalid.

Code inspection explains the mechanism. The LM Format Enforcer 0.11.3 free-text optimization caches tokens appropriate inside a string. A token that starts inside the string and also closes it can be excluded from that cache. The fallback traversal only explores tokens beginning with a quote, missing this punctuation-first transition.

Disabling the optimization in a test process allowed the complete Mistral sequence. Exhaustive Qwen enumeration was too slow and was stopped; the final saved test instead compared exact character-grammar acceptance with production token rejection for both tokenizers. No installed library or production decoder was changed.

At a string's hard character limit, the grammar permits only closure. This provides a coherent explanation for output continuing and then ending at 318–320 characters, sometimes mid-word. It is a confirmed mechanism, not yet a measurement of how many live judge failures a repair will remove.

**What this rules out:** the claim that all observed truncation is simply insufficient overall output-token budget. A larger budget does not make a suppressed token available.

**What it does not prove:** that the model would otherwise produce a correct judgment or faithful explanation.

### 10.2 Auditor test: change the evidence condition, hold the rest fixed

Four live Mistral calls used the same faculty-meeting caption, seed, prompt, schema and other claim fields. Only the supporting and contrary conditions changed.

| Probe | Supporting condition | Contrary condition | Reported support / conflict |
|---|---|---|---|
| Correct pair | Calm and relaxing | Chaotic and tense | true / false |
| Missing support | `None` | Chaotic and tense | true / false |
| Reversed pair | Chaotic and tense | Calm and relaxing | true / false |
| Identical pair | Calm and relaxing | Calm and relaxing | true / false |

All six audit Booleans were unchanged. The first four checks were true, support was true, and conflict was false. All responses had valid schemas and none hit the generation limit.

This proves failure of discrimination on these controlled inputs. The auditor falsely approves missing and reversed support and rejects a reasonable explicit opposite. Its explanatory prose can contradict its own Boolean assessment.

Both all-true and all-false audit sequences are permitted by the decoder, so there is no simple Boolean grammar rule forcing this answer pattern. More subtle decoder or prompt effects remain possible.

**What remains unknown:** how much comes from model capacity, quantization, schema conditioning or the wording of the audit task. These causes have not been separated by experiments. Replacing the model or rewriting the prompt would currently be a proposed intervention, not a proven fix.

### 10.3 Fault injection: can software contain a wrong auditor?

The test deliberately supplies a false-positive auditor and a schema-valid witness with support=`"None"` and conflict=`"None"`.

The production gate accepts it as `requirements_valid=True` with no errors.

The reason is deterministic: the fields are strings, `"None"` is nonempty, and the missing-evidence detector does not reject that placeholder. The gate depends on the model approval where an obvious structural rejection should already be possible.

This is not a test estimating how often a live model emits that exact combination. It proves that if it does, the current containment layer fails.

### 10.4 Round-trip test: an empty list becomes null

The hearing prompt uses a conversion equivalent to “use the value, otherwise write None.” An empty list is treated as absent and serialized as `None`. The witness reconstructs it as null rather than an empty list. The reconstructed object is sent to the audit without validating its input type against the core schema.

This is a confirmed interface defect. However, it cannot explain all semantic-audit failures, because the four live probes above used correctly typed empty lists and still failed.

### 10.5 Completeness test: punctuation is not grammar

The current saturation checker flags near-limit strings without terminal punctuation. Two targeted examples expose its limits:

- “because the” is not flagged when short, despite being unfinished.
- “The sky appears blue” is flagged at a 20-character ceiling, despite being a complete clause without punctuation.

The checker is useful as a narrow warning, but cannot certify that arbitrary explanations are complete. Removing it would hide genuine failures; treating it as a complete language validator would overstate its power.

### 10.6 What the test suite now means

The full suite ran 325 tests in 20.793 seconds: 321 passed and four were explicitly marked expected failures documenting the newly reproduced defects.

An expected failure means “this desired property is known not to hold yet.” It is not a repaired feature. Existing passing tests mainly establish software behaviour under their fixtures; they do not establish visual or semantic accuracy over real examples.

## 11. Consolidated problem list

| Priority | Problem | Evidence status | Consequence |
|---|---|---|---|
| Critical | Decoder suppresses legal string-closing tokens | Reproduced with both installed tokenizers; optimization isolated in Mistral diagnostic | Can force continuation and boundary truncation in structured outputs. |
| Critical | Caption auditor fails semantic discrimination | Four controlled live probes | Wrong requirements can be approved and useful opposites rejected. |
| Critical | Claim interpretation does not consistently govern evidence requirements | Saved idiom, products, normative and witness examples | Tribunal may investigate the wrong claim or condition. |
| Critical containment defect | Placeholder requirements can pass | Deterministic fault injection | An auditor error can become an accepted unusable requirement. |
| High | Typed lists are lost across prompt reconstruction | Deterministic reproduction | Audit input meaning and types can be corrupted between stages. |
| High | Relation direction is lost despite usable OCR | Products example and four missed contradictions | Correct observations can support a wrong final inference. |
| High | Judge repair does not remove decoder cause | Same decoder/schema used in repair; zero successful retries in run | Repeated work reproduces the bottleneck. |
| High | No demonstrated tribunal benefit | Ten unchanged predictions; zero corrections | Core research objective remains unmet. |
| High | Independent checks lack measured error rates | Architecture exists; annotation/qualification incomplete | Correlated errors and false rejection remain unquantified. |
| High | Explanation faithfulness unvalidated | Contradictory rationales; no completed human rating study | A detailed trace may explain the wrong relationship. |
| High for research | Confidence and acceptance policy uncalibrated | All final confidences 0.35; no fitted calibrator | Confidence cannot meaningfully distinguish these cases. |
| High for research | Adequate held-out and compute-matched comparisons missing | Current run is ten inspected development cases | No support for broad superiority or publication-ready gain claims. |
| Moderate | Metrics can be misread as stronger guarantees | 100% source citation coexists with 0% directional proof | “Grounded” and “valid” may mislead without qualification. |
| Moderate | High computation with no observed benefit | 249 generations, about 37 minutes | Efficiency must be assessed alongside any eventual gain. |
| Moderate | Dataset overlap/provenance work unfinished | Audited shared image groups; gated rebuild deferred | Requires disclosure and leakage-controlled evaluation. |
| Moderate | Documentation is partly stale | README versus current code | Readers may misunderstand review rounds and memory behaviour. |
| Moderate | First/repair raw-output preservation is incomplete | Saved attempts retain diagnostics, not a separate raw text for every attempt | Limits exact retry-cause analysis. |

## 12. What is working, or acceptable temporarily

There are useful foundations: stagewise model loading, completed artifact export, explicit rejected-proposal records, case-bound evidence hashes, protection of the raw caption, controlled reuse, and separate accounting of output failures and genuine abstentions.

Preserving a baseline answer when a proposal is unverified is reasonable temporary containment. It is not a satisfactory final tribunal if it always happens.

Feedback was disabled in this run, so zero feedback updates are expected. An empty feedback log is not evidence that the enabled feedback system failed.

The two invalid ABSTAIN outputs are not evidence that the judge is sensibly uncertain too often. The correct response is to repair output and semantic validity, not force more binary proposals.

The available evidence does not identify high-end hardware as the solution. A previous focused review replay stayed around 7.1 GB peak allocated memory without OOM recovery, while still failing explanation completeness. Consumer-GPU feasibility and semantic reliability are different questions.

## 13. Main learnings from the diagnosis

1. **More tokens cannot repair a blocked legal token.** Inspect the generation machinery before attributing every malformed answer to the model.
2. **Valid JSON is not valid reasoning.** A string can have the right type and the wrong meaning.
3. **A fresh auditor is not ground truth.** Measure whether it distinguishes controlled positive and negative conditions.
4. **Correct observations are insufficient without correct bindings.** Who has which property matters as much as recognizing the words.
5. **Figurative meaning needs an explicit connection to evidence.** A correct interpretation field is useless if the next stage reverts to literal requirements.
6. **Deterministic boundaries must contain obvious model failures.** Placeholders and malformed typed data should not rely on another model to be rejected.
7. **Safety and usefulness must be measured separately.** Zero harmful revisions with zero accepted revisions establishes little about tribunal quality.
8. **A gold-matching answer can still have bad reasoning.** Do not tune acceptance gates using hindsight about a debugging example.
9. **The failure trail is itself part of research quality.** Preserve unsuccessful attempts and identify which stage rejected what.
10. **Small controlled tests are more diagnostic than another large failing run.** The fixed-input auditor tests and token probe revealed causes hidden by aggregate accuracy.

## 14. Evidence-based repair order and completion criteria

This section specifies requirements for future work, not completed repairs.

### Phase 1: fix deterministic correctness boundaries

Repair legal token transitions while retaining constraints against illegal JSON. Test punctuation-plus-quote tokens, escapes, Unicode, whitespace, field changes, length limits and completion with both tokenizers. An exhaustive slow-path diagnostic is not automatically an acceptable implementation.

Preserve typed objects between stages. Reject null or placeholder content where a meaningful requirement is required, independent of any auditor approval. Turn the corresponding expected-failure tests into ordinary passing tests after actual repairs.

### Phase 2: qualify claim and witness semantics

Make expressed claim, interpretation hypothesis and evidence requirements explicitly consistent. Preserve roles, negation, quantities, degree, manner and scope. Use paired tests that alter one semantic property at a time.

The four auditor probes must distinguish missing, reversed, identical and correct states. Expand beyond the faculty-meeting example to avoid overfitting. Measure false approval and false rejection separately. Isolate prompt/schema/model effects before selecting a change.

### Phase 3: qualify judge and repair behaviour

Replay fixed saved cases with the corrected decoder. Save each attempt's raw output, effective prompt identity, budget and validation result. Check that the judge produces complete text, consistent labels/relations and source-preserving premises without merely copying a faulty prior response.

Judge output validity and semantic correctness must be reported separately. A formatting improvement is not yet an accuracy improvement.

### Phase 4: qualify verification and explanation faithfulness

Check every decisive relation against source evidence. Measure whether order-swapped verification agrees for the right reason. Test entity swaps, panel swaps, removed evidence and plausible alternatives. Obtain independent human ratings of claim preservation, image observation, citation support and explanation faithfulness.

### Phase 5: demonstrate research value

Freeze the system and acceptance policy, fit calibration on its separate partition, and compare against baseline, independent voting, self-consistency and conventional debate under declared compute conditions. Use image-group-aware statistics, sufficient sample sizes and prespecified seeds where applicable. Report corrections, harms, coverage, invalid-output rates, costs and uncertainty—not only final accuracy.

The existing research protocol's five-percentage-point improvement target is a planning objective, not a guaranteed outcome. If the system does not beat strong controls, the paper must report that honestly.

## 15. Documentation, attribution and research integrity

Some README statements reflect older code. It describes no second tribunal round and cache-free-first judge generation; current code permits up to two bounded reviews and starts generation with caching enabled. This report follows the code and saved run configuration, not those older statements.

The broader implementation tracker still contains 116 requested items. Implementation status is not the same as empirical completion. No statement in this report supersedes unfinished qualification with a success claim.

The project has an archived ten-paper research collection with forty hashed source files and attribution notes. The archive documents prior art and motivates design choices; it is not proof that these adaptations work in FigDebate. Saving a paper also does not grant permission to reproduce its copyrighted contents. The structured-decoder adaptation has separate MIT attribution.

The appropriate paper position today is: an evidence-oriented figurative tribunal under development, with documented engineering and semantic limitations—not an already validated superior reasoning system.

## 16. Final assessment

FigDebate has a substantial orchestration and audit structure, but the reliability of its central reasoning stages has not caught up with that structure.

The main observed sequence is:

**Unreliable claim requirements and auditing → weak or wrongly directed evidence → mostly invalid supervisor responses → rejected revisions → unchanged baseline answers.**

The decoder diagnosis adds an important shared low-level cause beneath those stages. It makes a specific engineering repair justifiable, but does not erase the separate semantic failures.

No finite test suite can guarantee correct language-model judgments for every possible input. What can be required without compromise is exact input handling, enforceable structural invariants, explicit uncertainty and failure states, preserved evidence, reproducible tests, and honest empirical qualification. Those guarantees must accompany—not be confused with—measured semantic accuracy.

## 17. Evidence index

Primary local sources used for this report:

- [Ten-case metrics](</C:/Users/Sai Prateek/Desktop/FigDebate_sent/FigDebate_main/outputs/checkpointed_supervisor_v5_oomfix_10/metrics_summary.txt>)
- [Ten-case configuration](</C:/Users/Sai Prateek/Desktop/FigDebate_sent/FigDebate_main/outputs/checkpointed_supervisor_v5_oomfix_10/run_config.json>)
- [Case-level predictions](</C:/Users/Sai Prateek/Desktop/FigDebate_sent/FigDebate_main/outputs/checkpointed_supervisor_v5_oomfix_10/predictions.csv>)
- [Detailed run records](</C:/Users/Sai Prateek/Desktop/FigDebate_sent/FigDebate_main/outputs/checkpointed_supervisor_v5_oomfix_10/records.jsonl>)
- [Root-cause diagnostic report](</C:/Users/Sai Prateek/Desktop/FigDebate_sent/FigDebate_main/docs/CRITICAL_ROOT_CAUSE_DIAGNOSIS_20260907.md>)
- [Controlled audit and boundary probe results](</C:/Users/Sai Prateek/Desktop/FigDebate_sent/FigDebate_main/outputs/critical_root_cause_20260907.json>)
- [Completed decoder comparison](</C:/Users/Sai Prateek/Desktop/FigDebate_sent/FigDebate_main/outputs/critical_decoder_probe_20260907_d.json>)
- [Known-defect regression reproductions](</C:/Users/Sai Prateek/Desktop/FigDebate_sent/FigDebate_main/tests/test_critical_diagnostic_reproductions.py>)
- [Implementation status and earlier qualifications](</C:/Users/Sai Prateek/Desktop/FigDebate_sent/FigDebate_main/docs/implementation_status.md>)
- [Qualification runbook](</C:/Users/Sai Prateek/Desktop/FigDebate_sent/FigDebate_main/docs/REPAIR_RUNBOOK.md>)
- [Method adaptations and attribution limits](</C:/Users/Sai Prateek/Desktop/FigDebate_sent/FigDebate_main/docs/METHOD_ADAPTATIONS.md>)

Current architecture was also checked directly in the stagewise runner, caption extractor, witness/audit modules, candidate generator, Arbiter scoring, tribunal state and acceptance code, independent review, runtime profiles and judge generation code. The report is descriptive; no production implementation was modified to produce it.
