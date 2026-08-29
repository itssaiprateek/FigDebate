# Independent multimodal judge

The judge is optional. Shadow and appellate use it after legacy debate; mediated
mode uses it as a label-blind planner. Tribunal mode uses a deterministic
pre-hearing audit and Qwen as the single bounded relation mediator after agent
testimony. It never replaces
Agent 1, Agent 2, the comparator, the Arbiter, the evidence verifier, or the
deterministic Review Board.

## Execution order

1. Agent 1 grounds visible image evidence with Qwen3-VL 4B Instruct through validated atomic questions.
2. Agent 2 extracts the caption claim with Mistral.
3. The deterministic comparator, evidence verifier, and Arbiter produce the
   established decision.
4. The evidence-constrained debate runs, preserving raw and gated proposals.
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
2. Escalated cases are routed to Qwen even when the ordinary debate router did
   not select them; they receive the same validated agent-response path.
3. Qwen sees the raw image, caption, structured agent records, comparator state,
   and the complete current evidence ledger. It does not see the current label
   or gold label.
4. Qwen returns a strict issue map with separate visual-agent questions,
   claim-agent questions, ledger citations, and verification requests.
5. The judge is unloaded before Agent 1 Qwen3-VL or Mistral is loaded for debate.
6. Each agent receives only the questions for its role. The provisional verdict,
   confidence, and mediator rationale are hidden to prevent anchoring.
7. New visual claims remain observations or candidates until an independent
   deterministic or multimodal verifier promotes their direction.
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
- cite independent decision-grade evidence with greater provenance-weighted
  reliability than the current direction; and
- pass the existing deterministic `review_revision` policy and final Review
  Board audit.

Failure of any condition preserves the established decision.

Mediated mode does not lower the global confidence threshold and never promotes
Qwen prose into the ledger. Its narrow verified-evidence tie-break can pass only
when all of the following hold:

- the mediation contract is valid and confidence is at least 0.80;
- Qwen's provisional direction matches the debated proposal;
- every cited ID belongs to the current sample and at least one citation is
  grounded current-image evidence;
- the claim contract is safe for directional reasoning;
- the visual reviewer supplies specific evidence for the proposed direction;
- the debate creates at least one independently verified decision-grade entry
  from targeted region verification; and
- the candidate still passes source attribution, review linkage, and final
  Review Board auditing.

Without these conditions, the original equal-evidence rejection remains.

## Tribunal execution order

1. The normal caption-blind grounding, claim audit, comparator, verifier, and
   initial Arbiter produce a checkpointed baseline decision.
2. A deterministic, label-blind pre-hearing audit records the structural issue
   and closes cases that already have adequate auditable evidence.
3. Escalated cases receive one issue-specific question per agent. No model vote
   is produced at Level 1.
4. Agent 1 returns a typed observation-only visual witness statement.
5. Agent 2 audits the immutable caption, polarity, roles, and opposing conditions.
6. The legacy debate Arbiter is not called in tribunal mode. Qwen reviews the
   original image, both responses, comparator candidates, and
   complete current ledger.
7. Qwen returns `RESOLVE`, `FOLLOW_UP`, or `ABSTAIN` under a strict JSON schema.
8. `FOLLOW_UP` creates at most one further neutral question for each agent.
9. A second review is the hard final round; another follow-up becomes abstention.
10. A resolved direction must cite either an existing matching decision-grade
    relation or a newly constructed three-source relation. The latter requires
    the current Agent 1 witness, Agent 2's valid claim audit, and Qwen's independent
    relation judgment; Qwen cannot promote its own statement by itself.
11. The deterministic Review Board compares reliability by independent
    provenance root and accepts only an adequately grounded revision.
12. Same-label confirmations are diagnostic only: they cannot mutate confidence
    or count as accepted revisions.
13. Confidence is capped by independently rooted verified evidence; derived
    restatements sharing a witness root are counted once.
14. The final record exports every decision checkpoint, round, stop reason, and
    accepted or rejected verification ID.

Questions containing entailment, contradiction, support, conflict, prediction,
verdict, or final-label wording are rejected before they reach an agent.

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
6. Test tribunal mode on the same IDs and report follow-up, abstention,
   correction, harm, verification, and runtime rates.
7. Promote only if paired accuracy is non-inferior and the accepted-revision
   harm rate is within the predeclared tolerance.

The local model is pinned as `Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Model weights remain ignored by
Git under `models/judge/`.

Download it on a new machine with:

    hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir models/judge/Qwen3.5-4B

For authenticated first-time downloads without committing a token:

    hf auth login
    hf auth whoami
