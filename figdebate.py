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
        hardware_profile="8gb",
        semantic_bridge_mode="disabled",
        global_seed=42,
        candidate_mode="independent",
        control_mode="none",
    ):
        from engine.orchestrator import Orchestrator

        self.orchestrator = Orchestrator(
            feedback_mode=feedback_mode,
            debate_mode=debate_mode,
            evidence_mode=evidence_mode,
            judge_mode=judge_mode,
            judge_scope=judge_scope,
            verified_feedback_path=verified_feedback_path,
            hardware_profile=hardware_profile,
            semantic_bridge_mode=semantic_bridge_mode,
            global_seed=global_seed,
            candidate_mode=candidate_mode,
            control_mode=control_mode,
        )

    def predict(self, image, caption):
        sample = {
            "image": image,
            "caption": caption,
        }

        return self.orchestrator.run_sample(sample)
