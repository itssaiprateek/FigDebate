from models.model_loader import LlavaModel
from models.mistral_loader import MistralModel

from agents.agent1 import VisualGroundingAgent
from agents.agent2 import ClaimExtractionAgent

from arbiter.arbiter import Arbiter

from phase4.feedback_loop import FeedbackLoop
from phase4.gpu_manager import GPUManager


class DebateEngine:

    def __init__(self):

        self.feedback_loop = FeedbackLoop()

        print("[Debate] Ready.")

    # -------------------------------------------------------
    # Decide whether debate is required
    # -------------------------------------------------------

    def should_debate(
        self,
        decision,
        threshold=0.65,
    ):

        if "debate_needed" in decision:
            return bool(decision.get("debate_needed", False))

        confidence = decision.get("confidence", 0)

        return confidence < threshold

    def run_debate(
        self,
        image,
        caption,
        visual_output,
        language_output,
        comparison,
        decision,
    ):

        agent1_prompt = self.build_agent1_challenge_prompt(
            visual_output,
            decision,
        )

        print("\nLoading LLaVA for debate...")

        llava = LlavaModel()

        agent1 = VisualGroundingAgent(llava)

        agent1_critique = agent1.critique(
            image,
            agent1_prompt,
        )

        del agent1
        del llava

        GPUManager.clear()

        agent2_prompt = self.build_agent2_challenge_prompt(
            language_output,
            decision,
        )

        print("\nLoading Mistral for debate...")

        mistral = MistralModel()

        agent2 = ClaimExtractionAgent(
            mistral.model,
            mistral.tokenizer,
        )

        arbiter = Arbiter(
            mistral.model,
            mistral.tokenizer,
        )

        agent2_critique = agent2.critique(
            caption,
            agent2_prompt,
        )

        decision = arbiter.analyze(
            caption,
            visual_output,
            language_output,
            comparison,
            agent1_critique=agent1_critique,
            agent2_critique=agent2_critique,
        )

        print("\nAgent 1:", agent1_critique)
        print("Agent 2:", agent2_critique)

        rounds = 2

        if self.agents_disagree(agent1_critique, agent2_critique):

            round3_prompt = self.build_round3_prompt(
                agent1_critique,
                agent2_critique,
            )

            agent2_critique = agent2.critique(
                caption,
                round3_prompt,
            )

            decision = arbiter.analyze(
                caption,
                visual_output,
                language_output,
                comparison,
                agent1_critique=agent1_critique,
                agent2_critique=agent2_critique,
            )

            rounds = 3

        try:
            del agent2
            del arbiter
            del mistral
        except:
            pass

        GPUManager.clear()

        return decision, rounds

    # -------------------------------------------------------
    # Existing feedback loop
    # -------------------------------------------------------

    def generate_feedback(
        self,
        visual_output,
        language_output,
        comparison,
        decision,
    ):

        return self.feedback_loop.generate_feedback(

            visual_output,

            language_output,

            comparison,

            decision,

        )

    def generate_feedback_batch(self, history):

        return self.feedback_loop.generate_feedback_batch(history)

    # -------------------------------------------------------
    # Prompt for Agent 1 (Visual Grounding)
    # -------------------------------------------------------

    def build_agent1_challenge_prompt(
        self,
        visual_output,
        decision,
    ):

        return f"""
You are the Visual Grounding Agent.

Your job is NOT to agree with the Arbiter.

Your job is to critically inspect the Arbiter's reasoning and find mistakes if they exist.

Visual Analysis

{visual_output}

Arbiter Decision

Label:
{decision["label"]}

Explanation:
{decision["explanation"]}

Based ONLY on the visual evidence,

either

ENDORSE

or

CHALLENGE

If you challenge, identify the specific flaw in the Arbiter's reasoning.

Return your answer in exactly this format:

Stance:
ENDORSE
or
CHALLENGE

Reason:
Explain your decision in 2-4 sentences.
"""

    # -------------------------------------------------------
    # Prompt for Agent 2 (Linguistic)
    # -------------------------------------------------------

    def build_agent2_challenge_prompt(
        self,
        language_output,
        decision,
    ):

        return f"""
You are the Linguistic Analysis Agent.

Your job is NOT to agree with the Arbiter.

Your job is to critically inspect the Arbiter's reasoning and find mistakes if they exist.

Caption Analysis

{language_output}

Arbiter Decision

Label:
{decision["label"]}

Explanation:
{decision["explanation"]}

Based ONLY on your linguistic reasoning,

either

ENDORSE

or

CHALLENGE

If you challenge, identify the specific flaw in the Arbiter's reasoning.

Return your answer in exactly this format:

Stance:
ENDORSE
or
CHALLENGE

Reason:
Explain your decision in 2-4 sentences.
"""

    # -------------------------------------------------------
    # Round 3 Trigger
    # -------------------------------------------------------

    def agents_disagree(
        self,
        agent1_response,
        agent2_response,
    ):

        return (

            agent1_response["stance"]

            !=

            agent2_response["stance"]

        )

    # -------------------------------------------------------
    # Round 3 Prompt
    # -------------------------------------------------------

    def build_round3_prompt(
        self,
        agent1_response,
        agent2_response,
    ):

        return f"""
Agent 1 Response

{agent1_response}

Agent 2 Response

{agent2_response}

The two agents disagree.

Reanalyse ONLY the disputed issue.

Do not repeat your previous analysis.

Focus only on resolving the disagreement.

Return your answer in exactly this format:

Stance:
ENDORSE
or
CHALLENGE

Reason:
Explain whether Agent 1's criticism changes your opinion.
"""