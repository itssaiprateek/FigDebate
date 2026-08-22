import json
import os
from datetime import datetime


class FeedbackLoop:

    def __init__(self,
                 max_examples=5,
                 log_file="step4_feedback_log.json"):

        self.max_examples = max_examples
        self.log_file = log_file

        self.agent1_memory = []
        self.agent2_memory = []

        print("[FeedbackLoop] Ready.")

    # --------------------------------------------------------
    # Failure Classification
    # --------------------------------------------------------

    def classify_failure(
        self,
        visual_output,
        language_output,
        comparison,
        decision
    ):

        alignment = comparison.get("alignment_score", 0)

        supporting = comparison.get(
            "supporting_points",
            [],
        )

        conflicting = comparison.get(
            "conflicting_points",
            [],
        )

        symbolic = str(
            visual_output.get(
                "symbolic_tone",
                ""
            )
        ).lower()

        figurative = str(
            language_output.get(
                "figurative_type",
                ""
            )
        ).lower()

        confidence = decision.get(
            "confidence",
            0
        )

        # -------------------------------
        # Agent1 hallucinated symbolism
        # -------------------------------

        hallucination_words = [

            "may represent",
            "could represent",
            "possibly",
            "symbolizes",
            "might symbolize",
            "perhaps",
            "appears to represent"

        ]

        for word in hallucination_words:

            if word in symbolic:

                return "visual_misinterpretation"

        # -------------------------------
        # Agent2 unsure about figurative
        # -------------------------------

        if figurative in [

            "",
            "unknown",
            "literal"

        ]:

            return "pragmatic_misunderstanding"

        # -------------------------------
        # Comparator weak but arbiter
        # predicts contradiction
        # -------------------------------

        if (

            alignment < 0.35

            and

            decision.get("label") == "CONTRADICTS"

            and

            confidence < 0.5

        ):

            return "incorrect_mapping"

        # -------------------------------

        if not supporting:

            return "incorrect_mapping"

        if conflicting:

            return "incorrect_mapping"

        return "general"

    # --------------------------------------------------------
    # Build Few-shot Example
    # --------------------------------------------------------

    def build_example(

        self,

        failure,

        visual_output,

        language_output,

        comparison,

        decision,

    ):

        visual = visual_output.get(
            "visual_description",
            ""
        )

        intended = language_output.get(
            "intended_meaning",
            ""
        )

        figurative = language_output.get(
            "figurative_type",
            ""
        )

        label = decision.get(
            "label",
            ""
        )

        confidence = decision.get(
            "confidence",
            0
        )

        # ====================================

        if failure == "visual_misinterpretation":

            return f"""
Example

Previous Visual Reasoning

{visual}

Correct Strategy

Only describe objects and relationships
that are directly visible.

Avoid speculative symbolism.

Why

Unsupported symbolism causes incorrect
multimodal alignment.
""".strip()

        # ====================================

        if failure == "pragmatic_misunderstanding":

            return f"""
Example

Previous Interpretation

Figurative Type:
{figurative}

Meaning:
{intended}

Correct Strategy

Reconsider whether another figurative
reading fits better.

Look for sarcasm, irony,
hyperbole and metaphor separately.

Why

Incorrect figurative interpretation
misleads downstream reasoning.
""".strip()

        # ====================================

        if failure == "incorrect_mapping":

            return f"""
Example

Previous Arbiter Decision

Label:
{label}

Confidence:
{confidence}

Correct Strategy

Absence of evidence
does NOT imply contradiction.

Predict CONTRADICTS only when
visual evidence actively conflicts
with the caption.

Why

Weak semantic alignment should
lead to low confidence,
not automatic contradiction.
""".strip()

        # ====================================

        return """
Example

Re-examine previous reasoning carefully.

Prefer observable evidence.

Avoid unsupported assumptions.
""".strip()
    # --------------------------------------------------------
    # Store Few-shot Example
    # --------------------------------------------------------

    def remember_example(self, agent, example):

        if not example:
            return

        if agent == "agent1":

            self.agent1_memory.append(example)

            if len(self.agent1_memory) > self.max_examples:
                self.agent1_memory.pop(0)

        elif agent == "agent2":

            self.agent2_memory.append(example)

            if len(self.agent2_memory) > self.max_examples:
                self.agent2_memory.pop(0)

    # --------------------------------------------------------
    # Save Feedback Log
    # --------------------------------------------------------

    def save_log(self, failure, agent, example):

        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "failure": failure,
            "example_added": example
        }

        if os.path.exists(self.log_file):

            try:

                with open(self.log_file, "r") as f:
                    data = json.load(f)

            except Exception:

                data = []

        else:

            data = []

        data.append(entry)

        with open(self.log_file, "w") as f:

            json.dump(data, f, indent=4)

    # --------------------------------------------------------
    # Build Prompt From Memory
    # --------------------------------------------------------

    def build_prompt(self, agent, instruction):

        if agent == "agent1":

            memory = self.agent1_memory

        else:

            memory = self.agent2_memory

        prompt = instruction

        if len(memory):

            prompt += "\n\n==============================\n"
            prompt += "Previous Mistakes to Avoid\n"
            prompt += "==============================\n\n"

            for i, example in enumerate(memory, 1):

                prompt += f"\nExample {i}\n"
                prompt += "-" * 40 + "\n"
                prompt += example
                prompt += "\n"

        return prompt

    # --------------------------------------------------------
    # Main Feedback Function
    # --------------------------------------------------------

    def generate_feedback(

        self,

        visual_output,

        language_output,

        comparison,

        decision,

    ):

        failure = self.classify_failure(

            visual_output,

            language_output,

            comparison,

            decision,

        )

        example = self.build_example(

            failure,

            visual_output,

            language_output,

            comparison,

            decision,

        )

        # -------------------------------------

        if failure == "visual_misinterpretation":

            self.remember_example(

                "agent1",

                example,

            )

            self.save_log(

                failure,

                "Agent1",

                example,

            )

        elif failure == "pragmatic_misunderstanding":

            self.remember_example(

                "agent2",

                example,

            )

            self.save_log(

                failure,

                "Agent2",

                example,

            )

        else:

            self.remember_example(

                "agent1",

                example,

            )

            self.remember_example(

                "agent2",

                example,

            )

            self.save_log(

                failure,

                "Agent1",

                example,

            )

            self.save_log(

                failure,

                "Agent2",

                example,

            )

        # -------------------------------------

        agent1_instruction = """
Re-analyze the image.

Use only directly observable visual evidence.

Avoid unsupported symbolic interpretations.

If earlier reasoning was weak, explicitly correct it.

Produce a revised visual analysis.
"""

        agent2_instruction = """
Re-analyze the caption.

Determine the most appropriate figurative interpretation.

Avoid unsupported assumptions.

If earlier reasoning was weak, explicitly correct it.

Produce a revised linguistic analysis.
"""

        agent1_prompt = self.build_prompt(

            "agent1",

            agent1_instruction,

        )

        agent2_prompt = self.build_prompt(

            "agent2",

            agent2_instruction,

        )

        return {

            "failure_type": failure,

            "agent1_feedback": agent1_prompt,

            "agent2_feedback": agent2_prompt,

        }

    def generate_feedback_batch(self, history):

        for sample in history:

            self.generate_feedback(

                sample["visual_output"],

                sample["language_output"],

                sample["comparison"],

                sample["decision"],

            )