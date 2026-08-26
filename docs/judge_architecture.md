# Independent multimodal judge

The judge is optional. Shadow and appellate use it after debate; mediated mode
uses it as a label-blind overseer inside the debate workflow. It never replaces
Agent 1, Agent 2, the comparator, the Arbiter, the evidence verifier, or the
deterministic Review Board.

## Execution order

1. Agent 1 grounds visible image evidence with LLaVA.
2. Agent 2 extracts the caption claim with Mistral.
3. The deterministic comparator, evidence verifier, and Arbiter produce the
   established decision.
4. The existing selective debate runs exactly as before.
5. All legacy GPU models are unloaded.
6. For escalated cases, Qwen3.5-4B receives the raw image, caption, compact agent
   packets, comparator output, debate critiques, and current evidence ledger.
7. Qwen returns the strict judge JSON contract.
8. In shadow mode the verdict is logged only. In appellate mode the
   deterministic gate decides whether the proposal is safe to accept.

The primary decision and gold label are omitted from the judge packet to reduce
anchoring and prevent label leakage.

## Mediated execution order

1. The established visual, claim, comparator, evidence-verifier, and initial
   Arbiter stages run unchanged.
2. Only cases already selected for debate are routed to Qwen.
3. Qwen sees the raw image, caption, structured agent records, comparator state,
   and the complete current evidence ledger. It does not see the current label
   or gold label.
4. Qwen returns a strict issue map with separate visual-agent questions,
   claim-agent questions, ledger citations, and verification requests.
5. Qwen is unloaded before LLaVA or Mistral is loaded for debate.
6. Each agent receives only the questions for its role. The provisional verdict,
   confidence, and mediator rationale are hidden to prevent anchoring.
7. New visual claims remain untrusted until the existing targeted region or
   structured reinspection verifier promotes them to decision-grade evidence.
8. The Arbiter receives the verified debate evidence and advisory issue list.
9. The deterministic Review Board applies the normal stronger-evidence rule or
   the narrow mediated tie-break described below.

The mediation contract is:

    {
      "status": "MEDIATE|ABSTAIN",
      "provisional_verdict": "ENTAILS|CONTRADICTS|ABSTAIN",
      "confidence": 0.0,
      "evidence_ids": ["VF001"],
      "issue": "unresolved issue",
      "agent1_question": "targeted image question",
      "agent2_question": "targeted claim question",
      "verification_request": "specific verification"
    }

## Strict output contract

    {
      "verdict": "ENTAILS|CONTRADICTS|ABSTAIN",
      "confidence": 0.0,
      "evidence_ids": ["VF001"],
      "visual_observations": ["direct image observation"],
      "reason": "short audit"
    }

Extra fields, prose outside the JSON object, invalid confidence, and unknown
labels make the judgment invalid. Unknown evidence IDs block an appellate
revision.

## Revision safety

Qwen-generated prose is never inserted into the evidence ledger as visual
proof. A disagreeing appellate judgment must:

- be contract-valid and non-abstaining;
- have confidence of at least 0.75;
- report a direct visual observation;
- cite only evidence from the current sample;
- cite decision-grade evidence matching its proposed direction;
- cite strictly more matching decision-grade items than exist for the current
  direction; and
- pass the existing deterministic `review_revision` policy and final Review
  Board audit.

Failure of any condition preserves the established decision.

Mediated mode does not lower the global confidence threshold and never promotes
Qwen prose into the ledger. It relaxes only an equal-count directional evidence
tie. That tie can pass when all of the following hold:

- the mediation contract is valid and confidence is at least 0.80;
- Qwen's provisional direction matches the debated proposal;
- every cited ID belongs to the current sample and at least one citation is
  grounded current-image evidence;
- the claim contract is safe for directional reasoning;
- the visual reviewer supplies specific evidence for the proposed direction;
- the debate creates at least one independently verified decision-grade entry
  from targeted region verification or structured visual reinspection; and
- the candidate still passes source attribution, review linkage, and final
  Review Board auditing.

Without these conditions, the original equal-evidence rejection remains.

Judge disagreements are recorded as human-or-gold verification candidates.
They never update procedural memory by themselves, so the feedback loop cannot
turn the judge's own prediction into a future pseudo-label.

## Rollout

1. Confirm baseline parity with `--judge-mode disabled`.
2. Run `--judge-mode shadow --judge-scope escalated` on the locked development
   sample and inspect judge agreement, abstention, invalid-output, runtime, and
   counterfactual correction/harm rates.
3. Do not tune on V-FLUTE test.
4. Test appellate mode on the same selected development IDs.
5. Test mediated mode on exactly the same IDs and compare its debate corrections,
   harms, invalid contracts, accepted tie-breaks, and runtime.
6. Promote only if paired accuracy is non-inferior and the accepted-revision
   harm rate is within the predeclared tolerance.

The local model is pinned as `Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Model weights remain ignored by
Git under `models/judge/`.

Download it on a new machine with:

    hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir models/judge/Qwen3.5-4B

For authenticated first-time downloads without committing a token:

    hf auth login
    hf auth whoami
