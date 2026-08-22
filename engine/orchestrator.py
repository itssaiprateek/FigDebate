from models.model_loader import LlavaModel
from models.mistral_loader import MistralModel
import time

from agents.agent1 import VisualGroundingAgent
from agents.agent2 import ClaimExtractionAgent

from comparators.comparator_v2 import compare
from arbiter.arbiter import Arbiter

from engine.gpu_manager import GPUManager
from engine.debate import DebateEngine
from engine.evidence_ledger import attach_evidence_audit, build_evidence_ledger
from engine.evidence_verifier import AtomicEvidenceVerifier, merge_verified_evidence
from engine.relation_schema import attach_claim_relation
from engine.review_board import attach_final_review


class Orchestrator:

    def __init__(self):

        print("=" * 70)
        print("FigDebate Engine Ready")
        print("=" * 70)

        self.debate = DebateEngine()
        self.prediction_count = 0
        self.feedback_history = []

        # Keep feedback disabled for baseline and paper evaluation. The
        # current loop has no ground-truth validation before it stores
        # "mistakes", so enabling it could reinforce model errors.
        self.feedback_enabled = False

    def run_sample(self, sample):

        image = sample["image"]
        caption = sample["caption"]

        ##########################################################
        # ROUND 1 : AGENT 1
        ##########################################################

        print("\n==============================")
        print("ROUND 1 : Agent 1")
        print("==============================")

        llava = LlavaModel()

        agent1 = VisualGroundingAgent(llava)

        agent1_instruction = """
Re-analyze the image.

Use only directly observable visual evidence.

Avoid unsupported symbolic interpretations.

If earlier reasoning was weak, explicitly correct it.

Produce a revised visual analysis.
"""

        visual_output = agent1.analyze(
            image,
            feedback=None,
        )

        print("\nUnloading LLaVA...")

        del agent1
        del llava

        GPUManager.clear()

        ##########################################################
        # ROUND 1 : AGENT 2
        ##########################################################

        print("\n==============================")
        print("ROUND 1 : Agent 2")
        print("==============================")

        mistral = MistralModel()

        agent2 = ClaimExtractionAgent(
            mistral.model,
            mistral.tokenizer,
        )

        evidence_verifier = AtomicEvidenceVerifier()
        self.debate.nli_verifier = evidence_verifier.nli
        arbiter = Arbiter(
            mistral.model,
            mistral.tokenizer,
            nli_verifier=evidence_verifier.nli,
        )

        agent2_instruction = """
Re-analyze the caption.

Determine the most appropriate figurative interpretation.

Avoid unsupported assumptions.

If earlier reasoning was weak, explicitly correct it.

Produce a revised linguistic analysis.
"""

        language_output = agent2.analyze(
            caption,
            feedback=None,
        )
        language_output = attach_claim_relation(language_output, caption)

        comparison_started = time.time()
        comparison = compare(
            visual_output,
            language_output,
            caption=sample["caption"],
        )
        comparison_seconds = time.time() - comparison_started

        evidence_ledger = build_evidence_ledger(
            visual_output, language_output, comparison
        )
        evidence_ledger, evidence_verification = evidence_verifier.verify(
            evidence_ledger, language_output, comparison
        )
        comparison = merge_verified_evidence(
            comparison, evidence_ledger, evidence_verification
        )

        decision = arbiter.analyze(
            caption,
            visual_output,
            language_output,
            comparison,
        )

        decision = attach_evidence_audit(decision, evidence_ledger)
        decision = attach_final_review(
            decision,
            evidence_ledger,
            language_output.get("claim_contract", {}),
        )

        initial_decision = attach_evidence_audit(
            decision.get("_primary_decision", decision), evidence_ledger
        )

        # -------------------------------
        # Logging variables
        # -------------------------------

        round1_confidence = initial_decision.get("confidence", 0)

        debate_triggered = False
        debate_assessment = self.debate.debate_assessment(decision, comparison)
        debate_trigger_reason = debate_assessment.get("reason")

        round2_confidence = None
        debate_rounds = 0

        ##########################################################
        # SHOULD WE DEBATE?
        ##########################################################

        if debate_trigger_reason:

            debate_triggered = True

            print("\n" + "=" * 70)
            print("STARTING DEBATE")
            print("=" * 70)

            print("\nUnloading Mistral before debate...")

            del agent2
            del arbiter
            del mistral

            GPUManager.clear()

            debate_started = time.time()
            decision, rounds = self.debate.run_debate(
                image=image,
                caption=caption,
                visual_output=visual_output,
                language_output=language_output,
                comparison=comparison,
                decision=decision,
                evidence_ledger=evidence_ledger,
                debate_assessment=debate_assessment,
            )
            debate_details = decision.get("_debate", {}) or {}
            if debate_details.get("recovered_visual_output"):
                visual_output = debate_details["recovered_visual_output"]
            if debate_details.get("recovered_comparison"):
                comparison = debate_details["recovered_comparison"]
            if debate_details.get("recovered_evidence_verification"):
                evidence_verification = debate_details[
                    "recovered_evidence_verification"
                ]
            evidence_ledger = decision.get(
                "_evidence_ledger", evidence_ledger
            )

            round2_confidence = decision.get("confidence", 0)
            debate_rounds = rounds
            debate_seconds = time.time() - debate_started
        else:
            debate_seconds = 0.0

        ##########################################################
        # FEEDBACK LOOP
        ##########################################################

        self.prediction_count += 1

        if self.feedback_enabled:
            self.feedback_history.append({
                "visual_output": visual_output,
                "language_output": language_output,
                "comparison": comparison,
                "decision": decision,
                "initial_decision": initial_decision,
                "evidence_ledger": evidence_ledger,
            })

        if self.feedback_enabled and self.prediction_count % 50 == 0:

            self.debate.generate_feedback_batch(
                self.feedback_history
            )

            self.feedback_history.clear()

            print(f"\nFeedback updated after {self.prediction_count} predictions.")

        ##########################################################
        # FREE EVERYTHING
        ##########################################################

        print("\nFreeing Agent 1 / LLaVA and Mistral resources...")

        try:
            del agent2
            del arbiter
            del mistral
        except:
            pass

        GPUManager.clear()

        ##########################################################

        return {

            "visual_output": visual_output,

            "language_output": language_output,

            "comparison": comparison,

            "evidence_ledger": decision.get("_evidence_ledger", evidence_ledger),

            "evidence_verification": evidence_verification,

            "initial_decision": initial_decision,

            "decision": decision,

            "debate_triggered": debate_triggered,

            "debate_trigger_reason": debate_trigger_reason,

            "debate_level": debate_assessment.get("level", 0),

            "debate_need_score": debate_assessment.get("score", 0),

            "debate_need_signals": debate_assessment.get("signals", []),

            "debate_rounds": debate_rounds,

            "debate_details": decision.get("_debate", {}),
            
            "round1_confidence": round1_confidence,

            "round2_confidence": round2_confidence,

            "timing": {
                "agent1_seconds": visual_output.get("_generation_seconds", 0.0),
                "agent2_seconds": language_output.get("_generation_seconds", 0.0),
                "comparator_seconds": comparison_seconds,
                "evidence_verifier_seconds": evidence_verification.get(
                    "seconds", 0.0
                ),
                "arbiter_primary_seconds": decision.get("_timing", {}).get("primary_seconds", 0.0),
                "binary_resolution_seconds": decision.get("_timing", {}).get("binary_resolution_seconds", 0.0),
                "debate_seconds": debate_seconds,
                "sample_inference_seconds": (
                    visual_output.get("_generation_seconds", 0.0)
                    + language_output.get("_generation_seconds", 0.0)
                    + comparison_seconds
                    + evidence_verification.get("seconds", 0.0)
                    + decision.get("_timing", {}).get("total_seconds", 0.0)
                    + debate_seconds
                ),
            },

        }
