class FigDebate:
    """
    Public phase-neutral API for the FigDebate system.

    The implementation currently wraps the most complete existing engine:
    Agent 1, Agent 2, Comparator V2, Arbiter, optional Debate, and Feedback.
    Historical phase-named modules remain in place for reproducibility.
    """

    def __init__(self):
        from engine.orchestrator import Orchestrator

        self.orchestrator = Orchestrator()

    def predict(self, image, caption):
        sample = {
            "image": image,
            "caption": caption,
        }

        return self.orchestrator.run_sample(sample)