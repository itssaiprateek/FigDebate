class FigDebate:
    """
    Public phase-neutral API for the FigDebate system.

    The API wraps the canonical engine: visual grounding, claim extraction,
    evidence comparison, arbitration, selective debate, and feedback memory.
    """

    def __init__(
        self,
        *,
        feedback_mode="disabled",
        debate_mode="enabled",
        evidence_mode="enabled",
        judge_mode="disabled",
        judge_scope="escalated",
        verified_feedback_path=None,
    ):
        from engine.orchestrator import Orchestrator

        self.orchestrator = Orchestrator(
            feedback_mode=feedback_mode,
            debate_mode=debate_mode,
            evidence_mode=evidence_mode,
            judge_mode=judge_mode,
            judge_scope=judge_scope,
            verified_feedback_path=verified_feedback_path,
        )

    def predict(self, image, caption):
        sample = {
            "image": image,
            "caption": caption,
        }

        return self.orchestrator.run_sample(sample)
