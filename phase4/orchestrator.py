from models.model_loader import LlavaModel
from models.mistral_loader import MistralModel

from agents.agent1 import VisualGroundingAgent
from agents.agent2 import ClaimExtractionAgent

from comparators.comparator_v2 import compare
from arbiter.arbiter import Arbiter

from phase4.gpu_manager import GPUManager
from phase4.debate import DebateEngine


class Orchestrator:

    def __init__(self):

        print("=" * 70)
        print("Phase 4 Orchestrator Ready")
        print("=" * 70)

        self.debate = DebateEngine()
        self.prediction_count = 0
        self.feedback_history = []

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

        agent1_feedback = self.debate.feedback_loop.build_prompt(
            "agent1",
            agent1_instruction,
        )

        visual_output = agent1.analyze(
            image,
            feedback=agent1_feedback,
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

        arbiter = Arbiter(
            mistral.model,
            mistral.tokenizer,
        )

        agent2_instruction = """
Re-analyze the caption.

Determine the most appropriate figurative interpretation.

Avoid unsupported assumptions.

If earlier reasoning was weak, explicitly correct it.

Produce a revised linguistic analysis.
"""

        agent2_feedback = self.debate.feedback_loop.build_prompt(
            "agent2",
            agent2_instruction,
        )

        language_output = agent2.analyze(
            caption,
            feedback=agent2_feedback,
        )

        comparison = compare(
            visual_output,
            language_output,
        )

        decision = arbiter.analyze(
            caption,
            visual_output,
            language_output,
            comparison,
        )

        # -------------------------------
        # Logging variables
        # -------------------------------

        round1_confidence = decision.get("confidence", 0)

        debate_triggered = False

        round2_confidence = None
        debate_rounds = 0

        ##########################################################
        # SHOULD WE DEBATE?
        ##########################################################

        if self.debate.should_debate(decision):

            debate_triggered = True

            print("\n" + "=" * 70)
            print("STARTING DEBATE")
            print("=" * 70)

            print("\nUnloading Mistral before debate...")

            del agent2
            del arbiter
            del mistral

            GPUManager.clear()

            decision, rounds = self.debate.run_debate(
                image=image,
                caption=caption,
                visual_output=visual_output,
                language_output=language_output,
                comparison=comparison,
                decision=decision,
            )

            round2_confidence = decision.get("confidence", 0)
            debate_rounds = rounds

        ##########################################################
        # FEEDBACK LOOP
        ##########################################################

        self.prediction_count += 1

        self.feedback_history.append({
            "visual_output": visual_output,
            "language_output": language_output,
            "comparison": comparison,
            "decision": decision,
        })

        if self.prediction_count % 50 == 0:

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

            "decision": decision,

            "debate_triggered": debate_triggered,

            "debate_rounds": debate_rounds,

            "round1_confidence": round1_confidence,

            "round2_confidence": round2_confidence,

        }